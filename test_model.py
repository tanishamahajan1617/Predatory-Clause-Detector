import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# 1. Asli Model Architecture (Ye wo dimaag hai jo aapne train kiya)
class ToSModel(nn.Module):
    def __init__(self, n_classes=9):
        super(ToSModel, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        pooled_output = outputs[1] # Pooled output le rahe hain
        return self.out(self.drop(pooled_output))

# 2. Setup Device & Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = ToSModel(n_classes=9) # Ab error nahi aayega

# Model ke weights load karein
model.load_state_dict(torch.load("tos_expert_model.pth"))
model.to(device)
model.eval()

# 3. Categories (Jo training mein use ki thi)
categories = [
    "Unilateral Change", "Content Removal", "Jurisdiction", 
    "Choice of Law", "Limitation of Liability", "Unilateral Termination",
    "Arbitration", "Contract by Using", "Other"
]

def predict_clause(text):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0] # Multi-label ke liye sigmoid
    
    return probs

# --- TEST KARTE HAIN ---
test_sentence = "We reserve the right to modify or terminate the service for any reason, without notice at any time."
results = predict_clause(test_sentence)

print("\n--- Model Analysis ---")
print(f"Sentence: {test_sentence}\n")

for cat, score in zip(categories, results):
    status = "🚩 PREDATORY" if score > 0.5 else "✅ SAFE"
    print(f"{status} | {cat}: {score:.4f}")