import torch
import torch.nn as nn
from torch.optim import AdamW
from model_setup import setup_ai_data  # Apki purani file se data mangwaya
from model_architecture import ToSModel

def train():
    # 1. Device Setup (GPU agar hai toh)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Working on: {device}")

    # 2. Load Data & Model
    # Note: Make sure your model_setup.py returns (train_loader, test_loader)
    train_loader, test_loader = setup_ai_data() 
    model = ToSModel(n_classes=9).to(device)

    # 3. Hyperparameters
    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = nn.BCEWithLogitsLoss() # Multi-label classification ke liye best!
    epochs = 3

    print("\n--- Training Started ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss/len(train_loader):.4f}")

    # 4. Save the Brain (For your Agent/Extension)
    torch.save(model.state_dict(), "tos_expert_model.pth")
    print("Model Saved Successfully! 💾")

if __name__ == "__main__":
    train()