import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer

# 1. PyTorch Dataset Class 
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
        # encode_plus ki jagah seedha tokenizer call karein
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            return_token_type_ids=False,
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def setup_ai_data():
    print("Loading Model-Ready Dataset...")
    df = pd.read_csv("data/model_ready_tos.csv")
    
    # ZAROORI: String labels ko asli List mein badalna
    df['target_labels'] = df['target_labels'].apply(ast.literal_eval)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_sentence'].tolist(), 
        df['target_labels'].tolist(), 
        test_size=0.2, 
        random_state=42
    )

    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # 2. Dataset Objects banana
    train_ds = ToSDataset(train_texts, train_labels, tokenizer, max_len=128)
    test_ds = ToSDataset(test_texts, test_labels, tokenizer, max_len=128)

    # 3. DataLoaders banana (Batches of 16)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    print(f"✅ Data Ready: {len(train_texts)} train, {len(test_texts)} test")
    return train_loader, test_loader

if __name__ == "__main__":
    t_loader, v_loader = setup_ai_data()
    # Check if it works
    sample_batch = next(iter(t_loader))
    print("Batch Input Shape:", sample_batch['input_ids'].shape)