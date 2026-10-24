import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import math
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. Dataset Placeholder
# -----------------------------------------------------------------------------
class TriageMultimodalDataset(Dataset):
    def __init__(self, data_dir='mimic-iv-ed-demo-2.2 synthetic/files', max_seq_len=30):
        self.max_seq_len = max_seq_len
        
        # Load and merge data
        print("Loading MIMIC-IV-ED data...")
        df = pd.read_csv(f"{data_dir}/triage.csv")
        
        # Preprocess target (5 classes: Acuity 1 to 5 -> labels 0 to 4)
        df = df.dropna(subset=['acuity'])
        self.targets = torch.tensor(df['acuity'].values - 1, dtype=torch.long)
        
        # Preprocess texts
        df['chiefcomplaint'] = df['chiefcomplaint'].fillna("unknown")
        print("Tokenizing text with bert-base-uncased...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokens = tokenizer(
            df['chiefcomplaint'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_seq_len,
            return_tensors='pt'
        )
        self.texts = tokens['input_ids']
        self.masks = tokens['attention_mask'].bool()
        
        # Preprocess vitals
        numeric_features = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
        df['pain'] = pd.to_numeric(df['pain'], errors='coerce')
        
        vitals_np = df[numeric_features].values
        
        # Impute and Scale
        imputer = SimpleImputer(strategy='median')
        vitals_np = imputer.fit_transform(vitals_np)
        
        scaler = StandardScaler()
        vitals_np = scaler.fit_transform(vitals_np)
        
        self.vitals = torch.tensor(vitals_np, dtype=torch.float32)
        
        self.num_samples = len(df)
        print(f"Loaded {self.num_samples} samples.")
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'mask': self.masks[idx],
            'vitals': self.vitals[idx],
            'label': self.targets[idx]
        }

# -----------------------------------------------------------------------------
# 2. Text Tower (Mini-Transformer Encoder)
# -----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.3, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MiniTransformerEncoder(nn.Module):
    def __init__(self, vocab_size=30000, d_model=256, nhead=4, num_layers=4, d_ff=1024, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_ff, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, src, src_key_padding_mask):
        # src: (batch_size, seq_len)
        # src_key_padding_mask: (batch_size, seq_len), boolean tensor where True means padded token
        
        # Convert True (valid) to False for PyTorch Transformer mask, and False (padded) to True
        # In our dataset mask: 1 = valid, 0 = pad. PyTorch expects True for padded tokens.
        padding_mask = ~src_key_padding_mask 

        x = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        # Transformer output: (batch_size, seq_len, d_model)
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Global Average Pooling across non-padded tokens
        mask_expanded = src_key_padding_mask.unsqueeze(-1).expand(output.size()).float()
        sum_embeddings = torch.sum(output * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero
        pooled_output = sum_embeddings / sum_mask
        
        return pooled_output

# -----------------------------------------------------------------------------
# 3. Tabular Tower (Dense MLP)
# -----------------------------------------------------------------------------
class TabularMLP(nn.Module):
    def __init__(self, input_dim=7, output_dim=256, dropout=0.3):
        super().__init__()
        # Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
        hidden_dim = 128
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 4. Late Fusion & Classification Head
# -----------------------------------------------------------------------------
class LateFusionMultimodalNet(nn.Module):
    def __init__(self, num_vitals=7, vocab_size=30000, num_classes=5, dropout=0.3):
        super().__init__()
        self.text_tower = MiniTransformerEncoder(
            vocab_size=vocab_size, d_model=256, nhead=4, num_layers=4, d_ff=1024, dropout=dropout
        )
        self.tabular_tower = TabularMLP(input_dim=num_vitals, output_dim=256, dropout=dropout)
        
        # Fusion Head: Concat (256 + 256 = 512) -> Dense (128) -> ReLU -> Dropout -> Linear (num_classes)
        self.fusion_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, text, mask, vitals):
        text_emb = self.text_tower(text, mask)
        vitals_emb = self.tabular_tower(vitals)
        
        fused = torch.cat((text_emb, vitals_emb), dim=1) # (batch_size, 512)
        logits = self.fusion_head(fused)
        return logits

# -----------------------------------------------------------------------------
# 5. Training Loop with Learning Rate Scheduler & Early Stopping
# -----------------------------------------------------------------------------
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_and_validate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data Loaders
    print("Preparing datasets...")
    full_dataset = TriageMultimodalDataset(data_dir='mimic-iv-ed-demo-2.2 synthetic/files')
    
    # 80/20 Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Model: Using bert-base-uncased vocab_size (30522)
    model = LateFusionMultimodalNet(num_vitals=7, vocab_size=30522, num_classes=5, dropout=0.3)
    model.to(device)
    
    # Optimizer & Scheduler & Loss
    epochs = 20
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    
    criterion = nn.CrossEntropyLoss()
    
    # Early Stopping params
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    best_model_path = 'best_multimodal_triage_net.pth'
    
    print("Starting training...")
    for epoch in range(epochs):
        # -- Training Phase --
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            texts = batch['text'].to(device)
            masks = batch['mask'].to(device)
            vitals = batch['vitals'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(texts, masks, vitals)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['text'].to(device)
                masks = batch['mask'].to(device)
                vitals = batch['vitals'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(texts, masks, vitals)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # -- Early Stopping Logic --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f" --> Saved new best model with Val Loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == "__main__":
    train_and_validate()
