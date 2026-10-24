import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. Dataset Class
# ---------------------------------------------------------
class SynapseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

# ---------------------------------------------------------
# 2. Main Training Function
# ---------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading SYNAPSE dataset...")
    df = pd.read_csv('SYNAPSE_An Expert Annotated Dataset of Patient symptoms and Demographics.csv')
    
    # Drop NAs
    df = df.dropna()
    
    # Format Text: Combine demographic + symptoms into a natural language sentence for the transformer
    df['combined_text'] = (
        "Patient is a " + df['Age'].astype(str) + " " + df['Gender'].astype(str) + 
        ". Condition duration: " + df['Duration'].astype(str) + 
        ". Severity: " + df['Severity'].astype(str) + 
        ". Symptoms: " + df['Symptoms'].astype(str)
    )
    
    # Map Labels: Doctor Consultation -> 1 (Emergency Care), OTC Drug -> 0 (Home Remedy)
    label_map = {'Doctor Consultation': 1, 'OTC Drug': 0}
    df['label'] = df['Final Recommendation'].map(label_map)
    
    # --- FIX CLASS IMBALANCE ---
    # There are ~126k Emergency vs ~4k Home Remedy. The model is just predicting Emergency for everything.
    # Let's undersample the majority class to balance the dataset.
    df_home = df[df['label'] == 0]
    df_emerg = df[df['label'] == 1].sample(n=len(df_home), random_state=42)
    df_balanced = pd.concat([df_home, df_emerg]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced Dataset Size: {len(df_balanced)} (Home Remedy: {len(df_home)}, Emergency: {len(df_emerg)})")
    
    X = df_balanced['combined_text'].values
    y = df_balanced['label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Init Tokenizer
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = SynapseDataset(X_train, y_train, tokenizer)
    val_dataset = SynapseDataset(X_val, y_val, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model (Sequence Classification head on top of DistilBERT)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    
    # Optimizer & Scheduler
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    loss_fn = nn.CrossEntropyLoss()
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, targets)
            
            total_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if step % 50 == 0:
                print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        print(f"\n--- Epoch {epoch+1} Validation ---")
        print(classification_report(val_targets, val_preds, target_names=['Home Remedy', 'Emergency Care']))
        
    print("Training complete! Saving model...")
    model.save_pretrained('./synapse_model')
    tokenizer.save_pretrained('./synapse_model')
    print("Model saved to ./synapse_model")

if __name__ == "__main__":
    main()
