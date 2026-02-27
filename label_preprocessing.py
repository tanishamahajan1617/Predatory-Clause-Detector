import pandas as pd
import numpy as np

def prepare_target_labels():

    df = pd.read_csv("data/cleaned_tos.csv")
    
    label_cols = ['a', 'ch', 'cr', 'j', 'law', 'ltd', 'ter', 'use', 'pinc']
    
    print("\n Total Unfair Clauses per Category:")
 
    print(df[label_cols].sum())

    df[label_cols] = df[label_cols].astype(int)   
  
    df['target_labels'] = df[label_cols].values.tolist()
    
    df['cleaned_sentence'].replace('', np.nan, inplace=True)
    df.dropna(subset=['cleaned_sentence'], inplace=True)
    print(f"Cleaned Sentence : {df['cleaned_sentence'].iloc[50]}")
    print(f"Target Array     : {df['target_labels'].iloc[50]}")
    df.drop(columns=label_cols, inplace=True)
    df.drop(columns=['sentence'], inplace=True) 
   
    output_path = "data/model_ready_tos.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Model-ready dataset saved to: {output_path}")

if __name__ == "__main__":
    prepare_target_labels()