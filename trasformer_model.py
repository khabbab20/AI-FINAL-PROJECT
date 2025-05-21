import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time

class SimpleTokenizer:
    def __init__(self, vocab=None):
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.bos_token = '<bos>'  # beginning of sentence
        self.eos_token = '<eos>'  # end of sentence
        
        if vocab is None:
            self.vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
            self.word2idx = {token: idx for idx, token in enumerate(self.vocab)}
        else:
            self.vocab = vocab
            self.word2idx = {token: idx for idx, token in enumerate(self.vocab)}
    
    def add_tokens(self, tokens):
        """Add new tokens to the vocabulary"""
        new_tokens = []
        for token in tokens:
            if token not in self.word2idx:
                new_tokens.append(token)
                
        self.vocab.extend(new_tokens)
        self.word2idx = {token: idx for idx, token in enumerate(self.vocab)}
        return len(new_tokens)
    
    def build_vocab_from_text(self, texts, min_freq=1):
        """Build vocabulary from a list of texts"""
        word_counts = {}
        
        # Count word frequencies
        for text in texts:
            for word in text.split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Add words that meet minimum frequency
        new_tokens = [word for word, count in word_counts.items() 
                     if count >= min_freq and word not in self.word2idx]
        
        self.add_tokens(new_tokens)
        return len(self.vocab)
    
    def encode(self, text, add_special_tokens=True):
        """Convert text to token IDs"""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
            
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
            
        return [self.word2idx.get(token, self.word2idx[self.unk_token]) for token in tokens]
    
    def decode(self, ids):
        """Convert token IDs to text"""
        tokens = [self.vocab[idx] if idx < len(self.vocab) else self.unk_token for idx in ids]
        
        # Remove special tokens
        cleaned_tokens = []
        for token in tokens:
            if token in [self.bos_token, self.eos_token, self.pad_token]:
                continue
            cleaned_tokens.append(token)
            
        return ' '.join(cleaned_tokens)
    
    def __len__(self):
        return len(self.vocab)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (tensor not considered a model parameter but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape
        q = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out(context)
        
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention with residual connection and normalization
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4, d_ff=1024, max_seq_length=512, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Token embedding and positional encoding
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Process through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Output prediction
        output = self.output_layer(x)
        
        return output

    def generate_text(self, start_tokens, max_length, tokenizer, temperature=1.0):
        self.eval()
        current_tokens = start_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                logits = self(current_tokens)
                next_token_logits = logits[:, -1, :] / temperature
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probabilities, 1)
                
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                if next_token.item() == tokenizer.word2idx.get(tokenizer.eos_token, 0):
                    break
                
        return tokenizer.decode(current_tokens[0].tolist())

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for text in texts:
            tokenized = self.tokenizer.encode(text)
            # Truncate if necessary
            if len(tokenized) > max_length:
                tokenized = tokenized[:max_length]
            # Pad sequence
            tokenized = tokenized + [self.tokenizer.word2idx[self.tokenizer.pad_token]] * (max_length - len(tokenized))
            self.examples.append(tokenized)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])

def create_attention_mask(batch, pad_token_id):
    """Create attention mask to prevent attending to padding tokens"""
    return (batch != pad_token_id).unsqueeze(1).unsqueeze(2)

def compute_perplexity(model, data_loader, tokenizer, device):
    """Compute perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            mask = create_attention_mask(input_ids, tokenizer.word2idx[tokenizer.pad_token])
            mask = mask.to(device)
            
            outputs = model(input_ids, mask)
            
            # Ignore padding tokens in loss calculation
            non_pad_mask = target_ids != tokenizer.word2idx[tokenizer.pad_token]
            num_tokens = non_pad_mask.sum().item()
            
            # Calculate loss
            loss = F.cross_entropy(
                outputs.contiguous().view(-1, len(tokenizer)),
                target_ids.contiguous().view(-1),
                ignore_index=tokenizer.word2idx[tokenizer.pad_token],
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += num_tokens
    
    # Calculate perplexity: exp(average_loss)
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def train_model(model, tokenizer, train_texts, val_texts=None, epochs=5, batch_size=16, 
                learning_rate=0.0001, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the transformer model"""
    # Prepare datasets
    train_dataset = TextDataset(train_texts, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_texts:
        val_dataset = TextDataset(val_texts, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Move model to device
    model = model.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'perplexity': []
    }
    
    # Training loop
    print(f"Training on {device}...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Create input and target sequences
            input_ids = batch[:, :-1]  # all tokens except the last one
            target_ids = batch[:, 1:]  # all tokens except the first one
            
            # Create attention mask
            mask = create_attention_mask(input_ids, tokenizer.word2idx[tokenizer.pad_token])
            mask = mask.to(device)
            
            # Forward pass
            outputs = model(input_ids, mask)
            
            # Compute loss (reshape for cross entropy)
            loss = torch.nn.functional.cross_entropy(
                outputs.contiguous().view(-1, len(tokenizer)),
                target_ids.contiguous().view(-1),
                ignore_index=tokenizer.word2idx[tokenizer.pad_token]
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        elapsed = time.time() - start_time
        
        # Validation
        val_loss = 0
        if val_texts:
            model.eval()
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    input_ids = batch[:, :-1]
                    target_ids = batch[:, 1:]
                    mask = create_attention_mask(input_ids, tokenizer.word2idx[tokenizer.pad_token])
                    mask = mask.to(device)
                    
                    outputs = model(input_ids, mask)
                    loss = torch.nn.functional.cross_entropy(
                        outputs.contiguous().view(-1, len(tokenizer)),
                        target_ids.contiguous().view(-1),
                        ignore_index=tokenizer.word2idx[tokenizer.pad_token]
                    )
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Calculate perplexity
            perplexity = compute_perplexity(model, val_loader, tokenizer, device)
            history['perplexity'].append(perplexity)
            
            # Adjust learning rate based on validation loss
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Time: {elapsed:.2f}s, Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.2f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs}, Time: {elapsed:.2f}s, Train Loss: {avg_train_loss:.4f}")
    
    return model, history

def generate_example(model, tokenizer, prompt, max_length=50, temperature=0.7):
    """Generate text from a prompt"""
    tokens = tokenizer.encode(prompt)
    input_tensor = torch.tensor([tokens]).to(next(model.parameters()).device)
    
    generated_text = model.generate_text(
        input_tensor, 
        max_length=max_length, 
        tokenizer=tokenizer, 
        temperature=temperature
    )
    
    return generated_text

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot perplexity
    if 'perplexity' in history and history['perplexity']:
        plt.subplot(1, 2, 2)
        plt.plot(history['perplexity'], label='Perplexity', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        plt.title('Validation Perplexity')
    
    plt.tight_layout()
    plt.show()

def save_model(model, tokenizer, filepath):
    """Save model and tokenizer"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'model_config': {
            'vocab_size': len(tokenizer),
            'd_model': model.token_embedding.embedding_dim,
            'num_heads': model.encoder_layers[0].self_attention.num_heads,
            'num_layers': len(model.encoder_layers),
            'd_ff': model.encoder_layers[0].feed_forward.linear1.out_features,
            'max_seq_length': model.positional_encoding.pe.size(1),
            'dropout': model.dropout.p
        }
    }
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load model and tokenizer"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(checkpoint['vocab'])
    
    # Create model
    model = CustomTransformer(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    # Sample text for demonstration
    example_texts = [
        "artificial intelligence is changing how we live and work",
        "deep learning models can process natural language",
        "transformers have revolutionized natural language processing",
        "neural networks are inspired by the human brain",
        "machine learning algorithms improve with more data",
        "data science combines statistics and programming",
        "computer vision systems can recognize objects in images",
        "reinforcement learning enables agents to learn from experience",
        "natural language understanding remains a challenging problem",
        "generative models can create realistic content"
    ]
    
    # Split into train and validation sets
    train_texts = example_texts[:8]
    val_texts = example_texts[8:]
    
    # Initialize tokenizer and build vocabulary
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab_from_text(example_texts)
    
    # Add more words to ensure a reasonable vocabulary size
    additional_words = """
    the and to of in for with on at by as from is was are were be been has have had could would 
    should might may will shall can must do does did using through before after during under 
    over between among within without above below beside behind beyond data model network 
    research science computer system learning algorithm transformer language processing 
    input output neural knowledge information technology future present past develop create
    build design implement deploy train test evaluate analyze understand generate predict
    recognize classify detect identify extract transform automate optimize improve enhance
    """.split()
    
    tokenizer.add_tokens(additional_words)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = CustomTransformer(
        vocab_size=len(tokenizer),
        d_model=128,      # Smaller for this example
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_length=128,
        dropout=0.1
    )
    
    # Train the model
    model, history = train_model(
        model,
        tokenizer,
        train_texts,
        val_texts=val_texts,
        epochs=20,
        batch_size=4,
        learning_rate=0.001,
        device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    save_model(model, tokenizer, "transformer_model.pt")
    
    # Generate text examples with different temperatures
    prompts = [
        "artificial intelligence",
        "deep learning",
        "neural networks"
    ]
    
    temperatures = [0.5, 0.7, 1.0]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        for temp in temperatures:
            generated = generate_example(model, tokenizer, prompt, max_length=20, temperature=temp)
            print(f"Generated (temp={temp}): {generated}")