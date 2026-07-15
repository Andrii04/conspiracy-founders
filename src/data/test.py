import urllib3.util
# Forziamo urllib3 (usata da requests) a dichiarare che l'IPv6 non è disponibile
urllib3.util.connection.HAS_IPV6 = False  

import requests

url_test = "https://arctic-shift.photon-reddit.com/api/users/interactions/users/list?author=gocoogs&before=1288195325&after=1280419325&limit=100"

print("--- TEST 2: FORZATURA IPV4 ---")
try:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url_test, headers=headers, timeout=60)
    print(f"Stato risposta: {response.status_code}")
    if response.status_code == 200:
        print("Successo! Forzando l'IPv4 la connessione è immediata.")
except Exception as e:
    print(f"Errore di connessione: {e}")