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
            print(f"[-] Errore di connessione per l'utente {username}: {e}")
            continue

        if response_user is None or len(response_user) == 0:
            print(f"[-] Utente {username} non trovato nelle API.")
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
            print(f"[-] Errore nel recupero dei subreddit per {username}: {e}")
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
            print(f"[+] Elaborati {counter}/{len(df_founders)} utenti... (Ultimo: {username}, Contributi Pre-Obs: {num_contributions})")
            
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
    
    print(f"\n[OK] Estrazione completata! Dataset salvato correttamente in: {output_path}")
    print(f"Dimensioni finali del file: {df_new.shape[0]} righe x {df_new.shape[1]} colonne.")


if __name__ == "__main__":
    # 1. CARICAMENTO DATASET (Con index_col=0 come richiesto)
    foundersFeatures = pd.read_csv("/home/andri/conspiracy-founders/data/interim/foundersFeatures_FINAL.csv", index_col=0)
    foundersData = pd.read_csv("/home/andri/conspiracy-founders/data/interim/foundersData_FINAL.csv", index_col=0)
    
    # =====================================================================
    # BARRIERA DI SICUREZZA: RIPRISTINO DELLE COLONNE SE "MANGIATE" DALL'INDEX
    # =====================================================================
    # Se 'username' è finito nell'indice a causa di index_col=0, lo riportiamo a colonna normale
    if 'username' not in foundersFeatures.columns and foundersFeatures.index.name == 'username':
        foundersFeatures = foundersFeatures.reset_index()
        
    # Facciamo lo stesso controllo simmetrico per foundersData e la colonna 'founder'
    if 'founder' not in foundersData.columns and foundersData.index.name == 'founder':
        foundersData = foundersData.reset_index()

    print(f"Righe iniziali - Features: {len(foundersFeatures)} | Data: {len(foundersData)}")
    
    # =====================================================================
    # 2. CALCOLO DELL'ANZIANITÀ PRIMA DELLA FOUNDING DATE
    # =====================================================================
    # Creiamo il dizionario di mappatura partendo dalla colonna 'founder'
    mapping_date = foundersData.set_index('founder')['subreddit_creation_date'].to_dict()
    
    # Mappiamo la data di creazione temporanea sulla colonna 'username' di foundersFeatures
    foundersFeatures['creation_date_temp'] = foundersFeatures['username'].map(mapping_date)
    
    # 6 mesi espressi in secondi Unix (6 mesi * 30 giorni * 24 ore * 3600 secondi)
    SIX_MONTHS_SECONDS = 6 * 30 * 24 * 60 * 60
    
    # =====================================================================
    # 3. APPLICAZIONE DEL FILTRO DI SBARRAMENTO (>= 6 Mesi di Vita)
    # =====================================================================
    # Sottrazione tra colonne numeriche Unix
    condizione_6mesi = (foundersFeatures['creation_date_temp'] - foundersFeatures['active since']) >= SIX_MONTHS_SECONDS
    
    # Escludiamo i valori nulli per sicurezza statistica
    condizione_valida = condizione_6mesi & foundersFeatures['active since'].notna()
    
    # Applichiamo il filtro alle feature e rimuoviamo la colonna di appoggio
    foundersFeatures_clean = foundersFeatures[condizione_valida].drop(columns=['creation_date_temp'])
    
    # =====================================================================
    # 4. SINCRONIZZAZIONE SPECULARE 1:1
    # =====================================================================
    # Allineiamo foundersData tenendo solo i founder presenti nel dataset pulito delle feature
    foundersData_clean = foundersData[foundersData['founder'].isin(foundersFeatures_clean['username'])]
    
    # =====================================================================
    # 5. REPORT FINALE E SALVATAGGIO
    # =====================================================================
    print("\n" + "="*50)
    print("        REPORT DI PULIZIA DATI INTEGRATO")
    print("="*50)
    print(f"Founder idonei rimasti (anzianità >= 6m): {len(foundersFeatures_clean)}")
    print(f"Founder scartati (anzianità < 6m o NaN):  {len(foundersFeatures) - len(foundersFeatures_clean)}")
    print("-" * 50)
    
    if len(foundersFeatures_clean) == len(foundersData_clean):
        print(" PASSED: I dataset sono perfettamente speculari e allineati 1:1!")
    else:
        print(f"⚠️ WARNING: Rilevata asimmetria! Features: {len(foundersFeatures_clean)} righe, Data: {len(foundersData_clean)} righe.")
    print("="*50)
    
    # Path di output per i file puliti
    path_features_out = "/home/andri/conspiracy-founders/data/interim/foundersFeatures_FINAL.csv"
    path_data_out = "/home/andri/conspiracy-founders/data/interim/foundersData_FINAL.csv"
    
    # Salviamo i dataset. index=True preserva la colonna d'indice numerica originaria
    foundersFeatures_clean.to_csv(path_features_out, index=True)
    foundersData_clean.to_csv(path_data_out, index=True)
    
    print(f"\n[OK] Pulizia completata con successo! File salvati in:")
    print(f" -> Features: {path_features_out}")
    print(f" -> Data:     {path_data_out}")

    # =====================================================================
    # 6. SANITY CHECK: VERIFICA ALLINEAMENTO E ORDINE REALE 1:1 SU DISCO
    # =====================================================================
    print("\n" + "="*50)
    print("   AVVIO SANITY CHECK: VERIFICA PROFILO TEMPORALE")
    print("="*50)

    # Ricarichiamo i file CSV appena salvati su disco per un controllo reale
    df_feat_check = pd.read_csv(path_features_out, index_col=0)
    df_data_check = pd.read_csv(path_data_out, index_col=0)

    # Ripristino standard delle colonne in caso di indicizzazione automatica
    if 'username' not in df_feat_check.columns and df_feat_check.index.name == 'username':
        df_feat_check = df_feat_check.reset_index()
    if 'founder' not in df_data_check.columns and df_data_check.index.name == 'founder':
        df_data_check = df_data_check.reset_index()

    # Per garantire l'ordine identico prima del test, ordiniamo df_data_check 
    # basandoci sull'esatta sequenza della colonna 'username' di df_feat_check
    df_data_check = df_data_check.set_index('founder').loc[df_feat_check['username']].reset_index()
    
    # Sovrascriviamo il file dei dati su disco con l'ordine garantito e corretto
    df_data_check.to_csv(path_data_out, index=True)

    # --- INIZIO DEI TEST DI INTEGRITÀ ---
    
    # TEST 1: Cardinalità (Numero di righe identico)
    lunghezza_identica = len(df_feat_check) == len(df_data_check)
    status_1 = "SUPERATO ✅" if lunghezza_identica else "FALLITO ❌"
    print(f"[TEST 1] Conteggio righe speculare:        {len(df_feat_check)} vs {len(df_data_check)} -> {status_1}")

    # TEST 2: Corrispondenza degli elementi (Stessi utenti presenti nei due file, set-level)
    stessi_utenti_set = set(df_feat_check['username']) == set(df_data_check['founder'])
    status_2 = "SUPERATO ✅" if stessi_utenti_set else "FALLITO ❌"
    print(f"[TEST 2] Intersezione utenti (Set Match):  {status_2}")

    # TEST 3: Ordine Posizionale Riga per Riga (I vettori coincidono in sequenza)
    if lunghezza_identica:
        # .values estrae gli array numpy nativi; .all() restituisce True solo se OGNI riga è identica
        ordine_perfetto = (df_feat_check['username'].values == df_data_check['founder'].values).all()
        status_3 = "PERFETTO ✅" if ordine_perfetto else "DISALLINEATO ❌"
        print(f"[TEST 3] Allineamento sequenziale (1:1):  {status_3}")
    else:
        ordine_perfetto = False
        print(f"[TEST 3] Allineamento sequenziale (1:1):  IMPOSSIBILE (Dimensioni diverse) ❌")

    # --- VERDETTO FINALE ---
    print("-" * 50)
    if lunghezza_identica and stessi_utenti_set and ordine_perfetto:
        print(" VERDETTO: I dataset su disco sono SPECULARI al 100%!")
        print(" Riga 'i' di Features e Riga 'i' di Data puntano allo stesso utente.")
        print(" File pronti per la ripartizione in Anchors e Candidates.")
    else:
        print(" 🛑 ERRORE CRITICO: Rilevato disallineamento nei file salvati.")
        print(" Verificare eventuali anomalie o duplicati rimasti nei sorgenti.")
    print("="*50)
