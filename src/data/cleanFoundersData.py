import pandas as pd

def cleanFoundersData():
    # 1. Carica il dataset
    df = pd.read_csv("data/interim/foundersData.csv", index_col=0)
    
    # 2. Ordina il dataframe in base alla data di creazione del subreddit (dal più vecchio al più recente)
    # Usiamo 'subreddit_creation_date' perché rappresenta il momento esatto dell'evento di fondazione
    df_sorted = df.sort_values(by='subreddit_creation_date', ascending=True)
    
    # 3. Rimuovi i duplicati basandoti sulla colonna 'founder'
    # keep='first' dice a Pandas di tenere solo la PRIMA riga che incontra per quel founder.
    # Poiché abbiamo ordinato in modo crescente, la prima riga sarà matematicamente il suo primo founding event.
    df_clean = df_sorted.drop_duplicates(subset=['founder'], keep='first')
    
    # 4. Opzionale: Resetta l'indice per avere una numerazione pulita da 0 a N senza salti
    df_clean = df_clean.reset_index(drop=True)

    
    return df_clean

# Per usarlo:
if __name__ == "__main__":
    foundersData = pd.read_csv("/home/andri/conspiracy-founders/data/interim/foundersData_clean.csv", index_col=0)
    foundersFeatures = pd.read_csv("/home/andri/conspiracy-founders/data/raw/realFounders_features_raw.csv", index_col=0)
    
    import pandas as pd

    # =====================================================================
    # 1. ASSICURIAMO L'UNICITÀ DI FOUNDERS DATA (Barriera di Sicurezza)
    # =====================================================================
    # Forza foundersData ad avere solo righe uniche basate sulla colonna 'founder'
    df_data_unique = foundersData.drop_duplicates(subset=['founder'], keep='first').reset_index(drop=True)

    # =====================================================================
    # 2. PULIZIA E ALLINEAMENTO DI FOUNDERS FEATURES
    # =====================================================================
    # Passo A: Tieni solo le righe il cui 'username' esiste nei founder ufficiali
    df_feat_filtered = foundersFeatures[foundersFeatures['username'].isin(df_data_unique['founder'])].copy()

    # Passo B: Elimina i 217 duplicati interni presenti in foundersFeatures
    df_feat_unique = df_feat_filtered.drop_duplicates(subset=['username'], keep='first').reset_index(drop=True)

    # =====================================================================
    # 3. CONTROLLO INCROCIATO SPECULARE (Doppia Blindatura)
    # =====================================================================
    # Se per assurdo qualche utente in foundersData non avesse le feature, 
    # lo scremiamo anche da lì per garantire il pareggio perfetto delle righe
    df_data_final = df_data_unique[df_data_unique['founder'].isin(df_feat_unique['username'])].reset_index(drop=True)
    df_feat_final = df_feat_unique[df_feat_unique['username'].isin(df_data_final['founder'])].reset_index(drop=True)

    # =====================================================================
    # 4. VERIFICA REALE DEI CONTATORI
    # =====================================================================
    print("--- VERIFICA FINALE ALLINEAMENTO ---")
    print(f"Righe totali in foundersData (Final):     {df_data_final.shape[0]}")
    print(f"Righe totali in foundersFeatures (Final): {df_feat_final.shape[0]}")

    if df_data_final.shape[0] == df_feat_final.shape[0]:
        print(" PASSED: I due dataset sono perfettamente allineati 1:1!")
    else:
        print(" WARNING: C'è ancora discrepanza, controlla i missing values.")

    # 1. FORZIAMO L'ORDINE CRONOLOGICO
    # Impostiamo temporaneamente 'username' come indice di df_feat_final,
    # usiamo .loc[] passando la sequenza esatta (e già ordinata) di df_data_final['founder'],
    # e infine riportiamo 'username' a colonna normale con .reset_index()
    df_feat_final = df_feat_final.set_index('username').loc[df_data_final['founder']].reset_index()

    # =====================================================================
    # 2. VERIFICA MATEMATICA DI SICUREZZA (RIGA PER RIGA)
    # =====================================================================
    # Questo comando controlla se OGNI SINGOLA RIGA coincide perfettamente tra i due vettori
    ordine_identico = (df_data_final['founder'] == df_feat_final['username']).all()

    print("--- VERIFICA ORDINE CRONOLOGICO ---")
    if ordine_identico:
        print(" PASSED: L'ordine delle righe coincide al 100% con la Founding Date!")
    else:
        print(" WARNING: L'ordine non coincide ancora. Controlla che non ci siano anomalie nei nomi.")

    print(df_data_final.tail(5))
    print(df_feat_final.tail(5))

        
        