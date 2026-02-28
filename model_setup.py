import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer
def setup_ai_data():
    print("Loading Model-Ready Dataset...")
    df = pd.read_csv("data/model_ready_tos.csv")
    
    # 1. Data ko Train (Seekhne ke liye) aur Test (Exam ke liye) mein baantna
    # Hum 80% data AI ko sikhane ke liye denge, aur 20% uska test lene ke liye
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['cleaned_sentence'].tolist(), 
        df['target_labels'].tolist(), 
        test_size=0.2, 
        random_state=42
    )
    print(f"\n Training Data: {len(train_texts)} sentences")
    print(f"Testing Data : {len(test_texts)} sentences")

    # 2. XLM-RoBERTa Tokenizer Load Karna
    print("\nDownloading XLM-RoBERTa Tokenizer from Hugging Face... (Please wait)")
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # 3. Ek Demo dekhte hain ki Tokenizer karta kya hai!
    sample_sentence = train_texts[0]
    print("\n*** Tokenization Magic Demo ***")
    print(f"Original Text : {sample_sentence}")
    
    # Text ko numbers mein convert karna
    tokens = tokenizer(sample_sentence, truncation=True, padding=True, max_length=128)
    print(f"AI Numbers (Input IDs) : {tokens['input_ids']}")

if __name__ == "__main__":
    setup_ai_data()