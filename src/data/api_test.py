import requests
import pandas as pd
import json
import time

# Disabilita IPv6 per evitare colli di bottiglia di routing su Windows
import urllib3.util
urllib3.util.connection.HAS_IPV6 = False

# Configurazione parametri globali
base_url = "https://arctic-shift.photon-reddit.com"
username = "gocoogs"
OBSERVATION_WINDOW_SECONDS = 3 * 30 * 24 * 60 * 60

def test_interactions_list():
    """Testa l'endpoint per ottenere la lista dettagliata delle interazioni di un utente."""
    endpoint = "/api/users/interactions/users/list"
    
    # 1. Recupero date (con fallback)
    before = 1288195325  # Fallback temporale (fine 2010)
    try:
        founders_df = pd.read_csv("data/interim/foundersData_FINAL.csv", index_col=0)
        user_row = founders_df[founders_df["founder"] == username]
        if not user_row.empty:
            before = int(user_row["subreddit_creation_date"].values[0])
            print(f"[INFO] Data trovata nel CSV per {username}: {before}")
    except Exception as e:
        print(f"[INFO] Impossibile leggere CSV ({e}). Utilizzo fallback temporale.")

    after = before - OBSERVATION_WINDOW_SECONDS
    params = {
        "author": username,
        "before": before,
        "after": after,
        "limit": 20
    }

    print(f"\n" + "="*60)
    print(f"--- TEST 1: {endpoint} ---")
    print(f"Parametri inviati: {params}")
    print("="*60)

    # 2. Esecuzione Chiamata API e Diagnostica
    try:
        start_time = time.time()
        response = requests.get(base_url + endpoint, params=params, timeout=120)
        elapsed = time.time() - start_time
        
        print(f"\nRISPOSTA RICEVUTA IN {elapsed:.2f} SECONDI")
        print(f"URL REALE: {response.url}")
        print(f"HTTP STATUS CODE: {response.status_code}")
        print("-"*60)
        
        
        print("\n--- CORPO DELLA RISPOSTA (Primi 1000 caratteri) ---")
        print(response.text[:1000])
        print("="*60 + "\n")

    except requests.exceptions.Timeout as t:
        print(f"\n[ERRORE] Timeout di connessione locale scattato! Il server non ha risposto entro i limiti. Dettaglio: {t}")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERRORE DI RETE] Impossibile completare la richiesta. Dettaglio: {e}")


def test_acquaintances_histograms():
    """Testa l'endpoint per ottenere il conteggio aggregato (l'istogramma) delle interazioni di un utente."""
    endpoint = "/api/users/interactions/users"
    
    # 1. Recupero date (con fallback)
    before = 1288195325
    try:
        founders_df = pd.read_csv("data/interim/foundersData_FINAL.csv", index_col=0)
        user_row = founders_df[founders_df["founder"] == username]
        if not user_row.empty:
            before = int(user_row["subreddit_creation_date"].values[0])
            print(f"[INFO] Data trovata nel CSV per {username}: {before}")
    except Exception as e:
        print(f"[INFO] Impossibile leggere CSV ({e}). Utilizzo fallback temporale.")

    after = before - OBSERVATION_WINDOW_SECONDS
    params = {
        "author": username,
        "before": before,
        "after": after,
        "limit": 100000
    }

    print(f"\n" + "="*60)
    print(f"--- TEST 2: {endpoint} ---")
    print(f"Parametri inviati: {params}")
    print("="*60)

    # 2. Esecuzione Chiamata API e Diagnostica
    try:
        start_time = time.time()
        response = requests.get(base_url + endpoint, params=params, timeout=120)
        elapsed = time.time() - start_time
        
        print(f"\nRISPOSTA RICEVUTA IN {elapsed:.2f} SECONDI")
        print(f"URL REALE: {response.url}")
        print(f"HTTP STATUS CODE: {response.status_code}")
        print("-"*60)
        
        print("\n--- CORPO DELLA RISPOSTA (Primi 1000 caratteri) ---")
        print(response.text[:1000])
        print("="*60 + "\n")

    except requests.exceptions.Timeout as t:
        print(f"\n[ERRORE] Timeout di connessione locale scattato! Dettaglio: {t}")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERRORE DI RETE] Impossibile completare la richiesta. Dettaglio: {e}")


def test_acquaintances_subreddits():
    """Testa l'endpoint per estrarre i subreddit in cui sono attivi i conoscenti dell'utente."""
    endpoint = "/api/users/interactions/subreddits"
    
    # 1. Recupero date (con fallback)
    before = 1288195325
    try:
        founders_df = pd.read_csv("data/interim/foundersData_FINAL.csv", index_col=0)
        user_row = founders_df[founders_df["founder"] == username]
        if not user_row.empty:
            # L'endpoint subreddits usa la data di creazione meno la finestra di osservazione
            before = int(user_row["subreddit_creation_date"].values[0]) - OBSERVATION_WINDOW_SECONDS
            print(f"[INFO] Data calcolata per {username}: {before}")
    except Exception as e:
        before = before - OBSERVATION_WINDOW_SECONDS
        print(f"[INFO] Impossibile leggere CSV ({e}). Utilizzo fallback temporale pre-calcolato: {before}")

    params = {
        "author": username,
        "before": before,
        "limit": 100000
    }

    print(f"\n" + "="*60)
    print(f"--- TEST 3: {endpoint} ---")
    print(f"Parametri inviati: {params}")
    print("="*60)

    # 2. Esecuzione Chiamata API e Diagnostica
    try:
        start_time = time.time()
        response = requests.get(base_url + endpoint, params=params, timeout=120)
        elapsed = time.time() - start_time
        
        print(f"\nRISPOSTA RICEVUTA IN {elapsed:.2f} SECONDI")
        print(f"URL REALE: {response.url}")
        print(f"HTTP STATUS CODE: {response.status_code}")
        print("-"*60)
        
        print("\n--- CORPO DELLA RISPOSTA (Primi 1000 caratteri) ---")
        print(response.text[:1000])
        print("="*60 + "\n")

    except requests.exceptions.Timeout as t:
        print(f"\n[ERRORE] Timeout di connessione locale scattato! Dettaglio: {t}")
    except requests.exceptions.RequestException as e:
        print(f"\n[ERRORE DI RETE] Impossibile completare la richiesta. Dettaglio: {e}")


if __name__ == "__main__":
    # Esegue i tre test in sequenza per l'utente Fauster
    test_interactions_list()
    time.sleep(2)  # Pausa precauzionale tra un test e l'altro
