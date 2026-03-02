import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import os

# 1. Dataset Class
class ToSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
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
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', return_token_type_ids=False,
            truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# 2. Model Architecture
class ToSModel(nn.Module):
    def __init__(self, n_classes=9):
        super(ToSModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        return self.out(self.drop(outputs[1]))

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running on: {device}")
    
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Model check
    if not os.path.exists("tos_expert_model.pth"):
        print("❌ Error: 'tos_expert_model.pth' nahi mili!")
        return

    model = ToSModel(n_classes=9)
    model.load_state_dict(torch.load("tos_expert_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Dataset check
    try:
        df = pd.read_csv("data/cleaned_tos.csv")
        print("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'cleaned_tos.csv' nahi mili. Kya naam sahi hai?")
        return

    # AUTO-FIX: Column names ki tension khatam
    # Maan rahe hain pehla column Text hai, aur baaki labels
    texts = df.iloc[:, 0].astype(str).to_list()
    labels = df.iloc[:, 1:].values
    
    # 20% Data for testing
    split_idx = int(len(df) * 0.8)
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]

    test_dataset = ToSDataset(test_texts, test_labels, tokenizer, max_len=128)
    test_loader = DataLoader(test_dataset, batch_size=16)

    all_preds = []
    all_targets = []

    print("📊 Predicting...")
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device)

            outputs = model(ids, mask)
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    binary_preds = (all_preds > 0.5).astype(int)

    categories = list(df.columns[1:]) # CSV se labels ke naam utha liye

    print("\n--- 📈 CLASSIFICATION REPORT ---")
    print(classification_report(all_targets, binary_preds, target_names=categories, zero_division=0))

if __name__ == "__main__":
    evaluate()