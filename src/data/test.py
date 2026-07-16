import requests
import pandas as pd
import json
import time
import os

# Forza IPv4
import urllib3.util
urllib3.util.connection.HAS_IPV6 = False

base_url = "https://arctic-shift.photon-reddit.com"
users_folder_path = "data/raw/FOUNDERS_DATA"
username = "silence7"
founder_folder_path = os.path.join(users_folder_path, f"{username}_data")

# File locali
comments_rec_path = os.path.join(founder_folder_path, f"{username}_commentsReceived.csv")
posts_comm_path = os.path.join(founder_folder_path, f"{username}_postsCommented.csv")
comments_path = os.path.join(founder_folder_path, f"{username}_comments.csv")

output_path = os.path.join(founder_folder_path, "interactionsCountLocalTest.csv")
original_api_path = os.path.join(founder_folder_path, f"{username}_interactionsCount.csv")

print(f"--- TEST DI COERENZA 100% (CORRETTO) PER: {username} ---\n")

if not os.path.exists(comments_path):
    print(f"[ERRORE] Manca il file dei commenti del founder.")
    exit()

all_usernames = []

# ==================== 1. INBOUND (Ricevuti) ====================
# Chi ha risposto direttamente al founder (ID dei commenti ricevuti)
if os.path.exists(comments_rec_path):
    df_rec = pd.read_csv(comments_rec_path)
    if not df_rec.empty and "author" in df_rec.columns:
        all_usernames.extend(df_rec["author"].dropna().tolist())

# ==================== 2. OUTBOUND (Inviati) ====================
# Analizziamo ogni singolo commento scritto dal founder per trovare il destinatario diretto
try:
    df_comments = pd.read_csv(comments_path)
    if not df_comments.empty and "parent_id" in df_comments.columns:
        
        # Dizionari di supporto per evitare doppie chiamate e mappare i dati
        post_authors_map = {}
        if os.path.exists(posts_comm_path):
            df_posts_comm = pd.read_csv(posts_comm_path)
            if not df_posts_comm.empty and "id" in df_posts_comm.columns and "author" in df_posts_comm.columns:
                post_authors_map = dict(zip(df_posts_comm["id"].astype(str), df_posts_comm["author"].astype(str)))

        # Separiamo i parent_id per tipo
        parent_comment_ids = []
        direct_post_replies = [] # Memorizza (comment_id, parent_post_id)

        for _, row in df_comments.iterrows():
            pid = str(row.get("parent_id"))
            if pid.startswith("t1_"):
                parent_comment_ids.append(pid[3:])
            elif pid.startswith("t3_"):
                direct_post_replies.append(pid[3:])

        # A. Gestione risposte dirette ai POST (t3_)
        for post_id in direct_post_replies:
            author = post_authors_map.get(post_id)
            if author:
                all_usernames.append(author)

        # B. Gestione risposte dirette ai COMMENTI (t1_) -> Recupero autori padri via API
        parent_comment_ids = list(set(parent_comment_ids))
        if parent_comment_ids:
            parent_comment_authors_map = {}
            batch_size = 100
            i = 0
            while i < len(parent_comment_ids):
                batch = parent_comment_ids[i : i + batch_size]
                ids_parameter = ",".join(batch)
                response = requests.get(f"{base_url}/api/comments/ids", params={"ids": ids_parameter}, timeout=30)
                if response.status_code == 200:
                    batch_data = response.json().get("data", [])
                    for obj in batch_data:
                        cid = obj.get("id")
                        c_author = obj.get("author")
                        if cid and c_author:
                            parent_comment_authors_map[str(cid)] = str(c_author)
                i += batch_size
                time.sleep(0.1)

            # Ora scorriamo i commenti t1_ scritti dal founder e aggiungiamo il rispettivo autore padre
            for _, row in df_comments.iterrows():
                pid = str(row.get("parent_id"))
                if pid.startswith("t1_"):
                    clean_pid = pid[3:]
                    author = parent_comment_authors_map.get(clean_pid)
                    if author:
                        all_usernames.append(author)

except Exception as e:
    print(f"[ERRORE] Durante l'elaborazione Outbound: {e}")

# ==================== 3. AGGREGAZIONE E CONFRONTO ====================
df_temp = pd.DataFrame(all_usernames, columns=["author"])
df_counts = df_temp["author"].value_counts().reset_index()
df_counts.columns = ["author", "count"]

# Salva il test
df_counts.to_csv(output_path, index=False)
print(f"[OK] File di test generato: {output_path}\n")

if os.path.exists(original_api_path):
    print("="*60)
    print("CONFRONTO DI VALIDAZIONE")
    print("="*60)
    df_api = pd.read_csv(original_api_path)
    
    set_api = set(zip(df_api["author"], df_api["count"]))
    set_local = set(zip(df_counts["author"], df_counts["count"]))
    
    matches = set_api.intersection(set_local)
    accuracy = (len(matches) / len(set_api)) * 100 if len(set_api) > 0 else 0
    
    print(f"Righe file API: {len(df_api)} | Righe file Locale: {len(df_counts)}")
    print(f"Match esatti (Autore + Conteggio): {len(matches)} / {len(set_api)} ({accuracy:.2f}%)")
    
    print("\n--- Primi 5 record (File API originale) ---")
    print(df_api.head(5).to_string(index=False))
    
    print("\n--- Primi 5 record (Nuovo File Locale di Test) ---")
    print(df_counts.head(5).to_string(index=False))
    print("="*60 + "\n")