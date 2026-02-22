import pandas as pd
import os

def generate_offline_dataset():
    print("API is blocking us (Error 400). Generating High-Quality Offline Dataset instead...")
    
    # Real-world legal clauses for our 3 Categories
    data = [
        # Label 2: Predatory/Unlawful (Unfair terms) - RED FLAGS
        {"clause_text": "We may share, sell, or rent your personal data to third parties without your explicit consent.", "label": 2, "source": "Mock_Data"},
        {"clause_text": "You waive your right to participate in a class action lawsuit against the company.", "label": 2, "source": "Mock_Data"},
        {"clause_text": "We reserve the right to terminate your account at any time for any reason without prior notice or explanation.", "label": 2, "source": "Mock_Data"},
        {"clause_text": "The company takes no responsibility for any data breaches, leaks, or loss of your private information.", "label": 2, "source": "Mock_Data"},
        {"clause_text": "By using this service, you grant us a perpetual, irrevocable, royalty-free license to sell all your uploaded content.", "label": 2, "source": "Mock_Data"},
        {"clause_text": "We can change pricing and charge your credit card without notifying you in advance.", "label": 2, "source": "Mock_Data"},
        
        # Label 1: Questionable (Borderline/Sneaky) - YELLOW FLAGS
        {"clause_text": "We may update these terms of service from time to time by posting them on our website.", "label": 1, "source": "Mock_Data"},
        {"clause_text": "Your data may be stored in servers located outside your home country.", "label": 1, "source": "Mock_Data"},
        {"clause_text": "We collect your precise location data while the app is in the background to improve our services.", "label": 1, "source": "Mock_Data"},
        {"clause_text": "Third-party analytics tools are used on our platform to track your usage behavior.", "label": 1, "source": "Mock_Data"},
        {"clause_text": "We may send you promotional emails based on your browsing history across other websites.", "label": 1, "source": "Mock_Data"},
        
        # Label 0: Safe/Compliant (Good practices) - GREEN FLAGS
        {"clause_text": "You can delete your account and all associated data at any time from the account settings.", "label": 0, "source": "Mock_Data"},
        {"clause_text": "We will notify you via email at least 30 days before making any material changes to these terms.", "label": 0, "source": "Mock_Data"},
        {"clause_text": "Your personal information is encrypted using industry-standard protocols during transmission.", "label": 0, "source": "Mock_Data"},
        {"clause_text": "We do not sell your personal data to advertisers or third-party data brokers.", "label": 0, "source": "Mock_Data"},
        {"clause_text": "You retain full ownership and intellectual property rights of the content you create.", "label": 0, "source": "Mock_Data"}
    ]
    
    # List ko Pandas Table (DataFrame) mein badalna
    df = pd.DataFrame(data)
    
    # 'data' folder banana (agar nahi hai)
    os.makedirs("data", exist_ok=True)
    
    # Data ko CSV mein save karna
    file_path = "data/tosdr_training_data.csv"
    df.to_csv(file_path, index=False)
    
    print("\n" + "="*50)
    print(f"SUCCESS! Created offline dataset with {len(df)} clauses.")
    print(f"Data saved to: '{file_path}'")
    print("\nData Distribution:")
    print(df['label'].value_counts())
    print("="*50)

if __name__ == "__main__":
    generate_offline_dataset()

