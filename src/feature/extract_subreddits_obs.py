import pandas as pd
import requests
import time
import json
import os

THREE_MONTHS_SECONDS = 3 * 30 * 24 * 60 * 60

matching_dict_path = "data/interim/matching_dict_optimized.json"
founders_data_path = "data/interim/foundersData_FINAL.csv"
save_path = "data/raw/founders_subreddits_obs_3m.csv"

# 1. Recupero la lista degli utenti necessari
with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
anchors = list(matching_dict.keys())
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
target_founders = set(anchors + candidates) # Uso un set per ricerche veloci

# 2. Carico il dataset principale per recuperare le date di creazione
df_founders = pd.read_csv(founders_data_path)

# Creiamo un dizionario velocissimo {username: founding_date}
founder_dates = {}
for row in df_founders.itertuples():
    if row.founder in target_founders:
        founder_dates[row.founder] = int(row.subreddit_creation_date)

def extract_subreddits_obs():
    foundersSubredditsPreobs = []
    counter = 0
    
    base_url = "https://arctic-shift.photon-reddit.com"
    subreddits_url = "/api/users/interactions/subreddits"
    full_url = base_url + subreddits_url

    print(f"Inizio estrazione istogrammi per la finestra di osservazione (3 mesi pre-fondazione) per {len(founder_dates)} utenti...")

    for username, founding_date in founder_dates.items():
        counter += 1
        
        # Impostiamo la finestra di osservazione esatta
        before_time = founding_date
        after_time = founding_date - THREE_MONTHS_SECONDS
        
        params = {
            "author" : username, 
            "limit" : 100000, 
            "before" : before_time, 
            "after" : after_time
        }

        try:
            response = requests.get(url=full_url, params=params)
            
            # Controllo se l'API va in errore (es. limite di richieste)
            if response.status_code == 200:
                data = response.json().get("data", [])
            else:
                print(f"[{response.status_code}] Errore API per {username}")
                data = []
                
        except Exception as e:
            print(f"Errore di connessione per {username}: {e}")
            data = []

        # Somma totale dei contributi nei 3 mesi pre-obs per controllo
        num_contributions = sum(item.get('count', 0) for item in data if isinstance(item, dict))

        # Prepariamo la riga del dataset
        row_data = {
            "author": username,
            "total_obs_contributions": num_contributions,
            "subreddits": data
        }
        
        foundersSubredditsPreobs.append(row_data)

        if counter % 10 == 0 or counter == 1:
            print(f"Elaborati {counter}/{len(founder_dates)} utenti. (Ultimo: {username}, Contributi: {num_contributions})")

        # Cortesia verso il server
        time.sleep(0.2)

    # Salvataggio sicuro
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(foundersSubredditsPreobs)
    df.to_csv(save_path, index=False)
    
    print(f"\n✅ Estrazione completata! File salvato in: {save_path}")

if __name__ == "__main__":
    extract_subreddits_preobs()

