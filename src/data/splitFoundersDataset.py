import pandas as pd
import numpy as np
import os

def splitDatasetByRows():
    # PERCORSI DEI FILE
    data_path = "data/interim/foundersData_FINAL.csv"
    scale_path = "data/external/conspiracy_svd_cossim_vector.csv"

    if not os.path.exists(data_path):
        data_path = "/home/andri/conspiracy-founders/data/interim/foundersData_FINAL.csv"
        scale_path = "/home/andri/conspiracy-founders/data/external/conspiracy_svd_cossim_vector.csv"

    # Caricamento dei file
    foundersData = pd.read_csv(data_path, index_col=0)
    conspiracy_vector = pd.read_csv(scale_path, index_col=0)

    # Barriera di sicurezza per l'indice
    if 'founder' not in foundersData.columns and foundersData.index.name == 'founder':
        foundersData = foundersData.reset_index()

    total_global_subs = len(conspiracy_vector)
    print(f"Dati caricati.")
    print(f" -> Subreddit totali nella scala globale: {total_global_subs}")
    print(f" -> Founder totali da smistare:            {len(foundersData)}\n")

    # TAGLIO FISICO DELLE RIGHE SULLA SCALA GLOBALE (ILOC SLICING)

    # Calcoliamo gli indici di riga esatti
    cutoff_top_25 = int(total_global_subs * 0.25)
    cutoff_bottom_50 = int(total_global_subs * 0.50)

    # Prendiamo le prime 25% delle righe (Top Conspiracy)
    top_25_subs = conspiracy_vector.iloc[:cutoff_top_25]
    
    # Prendiamo le ultime 50% delle righe (Bottom Mainstream)
    bottom_50_subs = conspiracy_vector.iloc[-cutoff_bottom_50:]

    # FILTRAGGIO DEI FOUNDER IN BASE AI SUBREDDIT DELLE RIGHE ESTRATTE
    # Gli Anchors sono i founder il cui sub compare nell'indice del top 25% delle righe
    df_anchors = foundersData[foundersData['subreddit_name'].isin(top_25_subs.index)].copy()
    
    # I Candidates sono i founder il cui sub compare nell'indice del bottom 50% delle righe
    df_candidates = foundersData[foundersData['subreddit_name'].isin(bottom_50_subs.index)].copy()


    # analisi split dataset
    
    print("="*60)
    print("      REPORT DETTAGLIATO: SPLIT GEOMETRICO DELLE RIGHE")
    print("="*60)
    print("QUOTE SUL CATALOGO GLOBALE SUBREDDIT:")
    print(f" -> Subreddit nel Top 25% delle righe:       {len(top_25_subs)}")
    print(f" -> Subreddit nel Bottom 50% delle righe:    {len(bottom_50_subs)}")
    print("-" * 60)
    print("RISULTATO SUI TUOI FOUNDER REALI:")
    print(f" -> Anchors generati (Top 25% global):       {len(df_anchors)} founder")
    print(f" -> Candidates generati (Bottom 50% global):  {len(df_candidates)} founder")
    print(f" -> Esclusi (Fascia grigia centrale o assenti): {len(foundersData) - len(df_anchors) - len(df_candidates)} founder")
    
    # Verifica di sicurezza: nessuna sovrapposizione possibile
    overlap = set(df_anchors['founder']).intersection(set(df_candidates['founder']))
    print("-" * 60)
    if len(overlap) == 0:
        print("Verifica mutua esclusività:                  SUPERATO ✅")
    else:
        print(f"⚠️ ERRORE: {len(overlap)} utenti si sovrappongono!")
    print("="*60)

    # salva files
    path_anchors_out = "data/interim/anchors_test.csv"
    path_candidates_out = "data/interim/candidates_test.csv"

    df_anchors.to_csv(path_anchors_out, index=True)
    df_candidates.to_csv(path_candidates_out, index=True)

    print(f"\n[OK] Split completato basandosi rigorosamente sulle righe del file CSV!")
    print(f" -> Salvale Anchor:    {path_anchors_out}")
    print(f" -> Salvato Candidate: {path_candidates_out}")

if __name__ == "__main__":
    True