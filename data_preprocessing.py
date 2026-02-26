import pandas as pd 
import re
def clean_data(text):
     if not isinstance(text, str):
        return ""
     text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
     text = re.sub(r'<.*?>', '', text)
     text = re.sub(r'\s+', ' ', text).strip()
     return text
    
def preprocess_data():

    df = pd.read_csv("data/terms_of_service.csv")

    print(f"Number of rows in df: {df.shape[0]}")
    print(f"Number of columns in df: {df.shape[1]}")
    print("Cleaning the text data... (Please wait)")
    df['cleaned_sentence'] = df['sentence'].apply(clean_data)

    output_path = "data/cleaned_tos.csv"
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_path}")
    print("="*50)

  
    print("languages in the dataset:")
    print(df['language'].value_counts())
  
   


    







if __name__ == "__main__":
   preprocess_data()

   