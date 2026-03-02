import torch.nn as nn
from transformers import XLMRobertaModel

class ToSModel(nn.Module):
    def __init__(self, n_classes=9):
        super(ToSModel, self).__init__()
        # 1. Base Brain: XLM-RoBERTa
        self.roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        # 2. Dropout: Overfitting se bachne ke liye
        self.drop = nn.Dropout(p=0.3)
        # 3. Classifier: 768 hidden units ko 9 labels mein map karna
        self.out = nn.Linear(self.roberta.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        pooled_output = outputs[1] # Sentence-level representation
        output = self.drop(pooled_output)
        return self.out(output)