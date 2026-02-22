import requests
import json
import pandas as pd
import os
import time

def scrape_full_tosdr_data(max_pages=20):
    print("Starting Deep Data Collection from ToS;DR API...")
    clauses_list = []
    
    # Hum 1 se lekar max_pages tak loop chalayenge
    for page in range(1, max_pages + 1):
        print(f"Fetching data from Page {page}...")
        
        # URL mein ?page= lagakar hum agle page par jaate hain
        url = f"https://api.tosdr.org/point/v1/?page={page}"
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                points = data.get('parameters', [])
                
                # Agar kisi page par data khatam ho jaye, toh loop rok do
                if not points:
                    print("No more data found. Stopping scraper.")
                    break
                    
                # Har point ko extract karna
                for item in points:
                    title = item.get('title', '')
                    status = item.get('status', 'neutral')
                    
                    if title:
                        # 2 = Predatory/Bad, 1 = Questionable, 0 = Safe/Good
                        if status in ['decline', 'bad']:
                            label = 2
                        elif status in ['neutral']:
                            label = 1
                        else:
                            label = 0
                            
                        clauses_list.append({
                            "clause_text": title,
                            "label": label,
                            "original_status": status,
                            "source": "ToS;DR API"
                        })
            else:
                print(f"Error on page {page}. Server says: {response.status_code}")
            
            # Professional Scraper Rule: Server par load na pade isliye 1 second ka gap (pause) lena
            time.sleep(1)
            
        except Exception as e:
            print(f"Network error occurred: {e}")
            break
            
    # Data ikattha hone ke baad usko CSV mein save karna
    if clauses_list:
        df = pd.DataFrame(clauses_list)
        os.makedirs("data", exist_ok=True)
        file_path = "data/tosdr_full_dataset.csv"
        df.to_csv(file_path, index=False)
        
        print("\n" + "="*40)
        print(f"SUCCESS! Downloaded {len(df)} real clauses.")
        print(f"Data saved to '{file_path}'")
        print("Data Distribution (Kitne Safe, Questionable aur Predatory hain):")
        print(df['label'].value_counts())
        print("="*40)
    else:
        print("Failed to scrape any data.")

if __name__ == "__main__":
    # Hum API ke pehle 20 pages scrape kar rahe hain. 
    # Aap chaho toh is number ko badha kar 50 ya 100 bhi kar sakti ho!
    scrape_full_tosdr_data(max_pages=20)


