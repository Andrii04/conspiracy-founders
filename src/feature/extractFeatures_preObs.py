import os
import time
import requests
import pandas as pd
import numpy as np

def extract_features_pre_obs():
    foundersFeatures = []
    counter = 0
    base_url = "https://arctic-shift.photon-reddit.com/"
    users_url = "/api/users/search"
    subreddits_url = "/api/users/interactions/subreddits"

    # 1. CARICAMENTO DATASET DEI FOUNDER
    # Utilizziamo il percorso del dataset dei founder pulito precedentemente
    founders_data_path = "data/interim/foundersData_FINAL.csv"
    if not os.path.exists(founders_data_path):
        # Fallback nel caso stessi eseguendo con il percorso assoluto del tuo sistema
        founders_data_path = "/home/andri/conspiracy-founders/data/interim/foundersData_FINAL.csv"
        
    df_founders = pd.read_csv(founders_data_path)
    print(f"Caricati {len(df_founders)} founder da analizzare.")

    # 3 mesi espressi in secondi (3 mesi * 30 giorni * 24 ore * 60 minuti * 60 secondi)
    THREE_MONTHS_SECONDS = 3 * 30 * 24 * 60 * 60

    print("Inizio estrazione delle feature per il periodo di Pre-Observation...\n")

    # Ciclo su tutti i record del dataset usando itertuples (molto veloce)
    for row in df_founders.itertuples():
        username = row.founder
        founding_date = row.subreddit_creation_date
        
        # CALCOLO SBARRAMENTO TEMPORALE (6 mesi prima del founding event)
        cutoff_time = int(founding_date - THREE_MONTHS_SECONDS)

        # 2. CHIAMATA API 1: Anzianità Globale dell'Account
        params_user = {
            "author" : username,
            "limit" : 1
        }
        url_user = base_url + users_url
        
        try:
            response_user = requests.get(url_user, params=params_user).json().get("data")
        except Exception as e:
            print(f"Errore di connessione per l'utente {username}: {e}")
            continue

        if response_user is None or len(response_user) == 0:
            print(f"Utente {username} non trovato nelle API.")
            continue
            
        responseData = response_user[0]
        founderMetaData = responseData.get("_meta")
        
        c_time = founderMetaData.get("earliest_comment_at")
        p_time = founderMetaData.get("earliest_post_at")
        
        if c_time is not None and p_time is not None:
            active_since = min(c_time, p_time)
        elif c_time is not None:
            active_since = c_time
        elif p_time is not None:
            active_since = p_time
        else:
            active_since = np.nan

        # 3. CHIAMATA API 2: Subreddit filtrati con vincolo temporale 'before'
        # Passiamo il cutoff_time per bloccare l'aggregazione ai 6 mesi pre-fondazione
        params_subs = {
            "author" : username,
            "limit" : 1000000,
            "before" : cutoff_time 
        }
        url_subs = base_url + subreddits_url
        
        try:
            response_subs = requests.get(url_subs, params=params_subs).json()
            founderSubredditsRaw = response_subs.get("data")
        except Exception as e:
            print(f"Errore nel recupero dei subreddit per {username}: {e}")
            founderSubredditsRaw = []

        founderSubredditsClean = founderSubredditsRaw if founderSubredditsRaw else []

        # 4. CALCOLO DEI CONTRIBUTI FILTRATI (Somma dei count pre-observation)
        # Poiché l'API ha applicato il 'before', la somma dei count restituisce solo i post passati
        num_contributions = sum(int(item.get('count', 0)) for item in founderSubredditsClean if isinstance(item, dict))

        # Creazione del dizionario coerente con la struttura originale richiesta
        founderData = {
            "username" : username,
            "active since" : active_since,
            "num contributions" : num_contributions,
            "subreddits" : founderSubredditsClean
        }

        foundersFeatures.append(founderData)
        counter += 1
        
        # Stampiamo lo stato di avanzamento ogni 50 utenti per monitorare lo script
        if counter % 50 == 0 or counter == 1:
            print(f"Elaborati {counter}/{len(df_founders)} utenti (Ultimo: {username}, Contributi Pre-Obs: {num_contributions})")
            
        # Delay di cortesia per rispettare i limiti del server Photon/Arctic Shift
        time.sleep(0.15)

    # 5. SALVATAGGIO DEL DATASET GENERATO
    df_new = pd.DataFrame(foundersFeatures)
    
    # Manteniamo l'esatto ordine delle colonne richiesto dal vecchio dataset
    df_new = df_new[["username", "active since", "num contributions", "subreddits"]]
    
    output_path = "data/interim/foundersFeatures_preObs_3m.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salviamo sovrascrivendo l'intero file per garantire l'allineamento a 2248 righe pulite
    df_new.to_csv(output_path, index=False)
    
    print(f"\nEstrzione completata dfataset salvato correttamente in: {output_path}")
    print(f"Dimensioni finali del file: {df_new.shape[0]} righe x {df_new.shape[1]} colonne.")


if __name__ == "__main__":
    True
