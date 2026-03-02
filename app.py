from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel

# 1. FastAPI Initialize
app = FastAPI(title="Predatory Clause Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# 3. Model & Tokenizer Load
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = ToSModel(n_classes=9)

# Map location zaroori hai agar model GPU par train hua aur CPU par load ho raha ho
model.load_state_dict(torch.load("tos_expert_model.pth", map_location=device))
model.to(device)
model.eval()

# 4. Request Model (Naam match hona chahiye!)
class AnalysisRequest(BaseModel):
    text: str

# 5. API Endpoint
@app.post("/analyze")
async def analyze_text(request: AnalysisRequest): # Yahan ClauseRequest ki jagah AnalysisRequest kar diya
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device) # 128 ko 512 karein
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    # --- 0.3 Threshold Approach ---
    threshold = 0.15
    findings = []
    
    categories = [
        "Unilateral Change", "Content Removal", "Jurisdiction", 
        "Choice of Law", "Limitation of Liability", "Unilateral Termination",
        "Arbitration", "Contract by Using", "Other"
    ]

    for i, score in enumerate(probs):
        if score > threshold:
            findings.append({
                "category": categories[i],
                "probability": round(float(score), 4),
                "risk_level": "High" if score > 0.7 else "Medium"
            })

    return {
        "status": "success",
        "findings": findings
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)