import pandas as pd
import os
from datasets import load_dataset

def load_hf_dataset():
   print("Loading Hugging Face dataset...")
   try:
        dataset = load_dataset("joelniklaus/online_terms_of_service")
        df = pd.DataFrame(dataset['train'])
        print("Dataset loaded successfully.")

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/terms_of_service.csv", index=False)
        print("Dataset saved to data/terms_of_service.csv")

   except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")



if __name__ == "__main__":
    load_hf_dataset()





   