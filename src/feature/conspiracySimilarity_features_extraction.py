import pandas as pd
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# CONFIGURAZIONE

users_folder_path = "data/raw/FOUNDERS_DATA/"
matching_dict_path = "data/interim/matching_dict_optimized.json"

PROF_COMMENTS_PATH = "data/external/top_comments_per_subreddit.json"
PROF_SUBMISSIONS_PATH = "data/external/top_submissions_per_subreddit.json"

# File JSON di output finale
output_json_path = "data/raw/user_conspiracy_similarity.json"

# FUNZIONI DI SUPPORTO

def calculate_cosine_similarity(vec_a, vec_b):
    """Calcola la somiglianza del coseno tra due vettori numpy."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def load_founders_list():
    """Carica la lista completa dei founder dal dizionario di matching."""
    with open(matching_dict_path, "r") as f:
        matching_dict = json.load(f)
    anchors = list(matching_dict.keys())
    candidates = [candidate["candidate"] for candidate in matching_dict.values()]
    return list(set(anchors + candidates))

# ATTIVITÀ PRINCIPALE

if __name__ == "__main__":
    founders = load_founders_list()
    print(f"Caricati {len(founders)} founder da elaborare.")

    # 1. Caricamento del modello Sentence Transformer richiesto
    print("Caricamento del modello HuggingFace all-mpnet-base-v2...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # 2. Creazione dei documenti e degli embedding per i Subreddit del professore
    print("Costruzione dei vettori di riferimento per i subreddit conspiracy...")
    subreddit_texts = {}

    # Lettura dei commenti del professore
    if os.path.exists(PROF_COMMENTS_PATH):
        with open(PROF_COMMENTS_PATH, "r") as f:
            prof_comments = json.load(f)
        for sub_name, comments_list in prof_comments.items():
            text_block = " ".join([str(c.get("body", "")) for c in comments_list if isinstance(c, dict)])
            subreddit_texts[sub_name] = subreddit_texts.get(sub_name, "") + " " + text_block

    # Lettura delle submissions del professore
    if os.path.exists(PROF_SUBMISSIONS_PATH):
        with open(PROF_SUBMISSIONS_PATH, "r") as f:
            prof_submissions = json.load(f)
        for sub_name, subs_list in prof_submissions.items():
            text_block = " ".join([str(s.get("title", "")) + " " + str(s.get("selftext", "")) for s in subs_list if isinstance(s, dict)])
            subreddit_texts[sub_name] = subreddit_texts.get(sub_name, "") + " " + text_block

    # Generazione degli embedding per ogni subreddit
    subreddit_embeddings = {}
    for sub_name, full_text in subreddit_texts.items():
        if full_text.strip():
            subreddit_embeddings[sub_name] = model.encode(full_text.strip(), convert_to_numpy=True)

    print(f"Generati vettori di benchmark per {len(subreddit_embeddings)} subreddit conspiracy.")

    # 3. Creazione documenti utenti e calcolo della somiglianza media
    print("\nInizio elaborazione documenti e calcolo somiglianze per i founder...")
    conspiracy_similarity_results = {}

    for username in founders:
        folder = os.path.join(users_folder_path, f"{username}_data")
        posts_path = os.path.join(folder, f"{username}_posts.csv")
        comments_path = os.path.join(folder, f"{username}_comments.csv")

        user_text_fragments = []

        # Estrazione e unione dei post dell'utente
        if os.path.exists(posts_path):
            df_p = pd.read_csv(posts_path)
            if "title" in df_p.columns and "selftext" in df_p.columns:
                titles = df_p["title"].dropna().astype(str).tolist()
                bodies = df_p["selftext"].dropna().astype(str).tolist()
                user_text_fragments.extend(titles + bodies)

        # Estrazione e unione dei commenti dell'utente
        if os.path.exists(comments_path):
            df_c = pd.read_csv(comments_path)
            if "body" in df_c.columns:
                comments = df_c["body"].dropna().astype(str).tolist()
                user_text_fragments.extend(comments)

        user_document = " ".join(user_text_fragments).strip()

        # Se l'utente non ha testo valido, viene assegnato punteggio 0.0
        if not user_document:
            print(f"Mancanza di testo per l'utente {username}, score impostato a 0.0")
            conspiracy_similarity_results[username] = 0.0
            continue

        # Generazione del vettore per l'utente corrente
        user_vector = model.encode(user_document, convert_to_numpy=True)

        # Calcolo della somiglianza del coseno contro ciascun subreddit del professore
        similarities = []
        for sub_name, sub_vector in subreddit_embeddings.items():
            sim = calculate_cosine_similarity(user_vector, sub_vector)
            similarities.append(sim)

        # Calcolo della somiglianza coseno media finale
        mean_similarity = float(np.mean(similarities)) if similarities else 0.0
        conspiracy_similarity_results[username] = mean_similarity
        
        print(f"Utente {username} elaborato. Cosine Similarity Media: {mean_similarity:.4f}")

    # 4. Salvataggio finale del dizionario in formato JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(conspiracy_similarity_results, f, indent=4)

    print(f"\nElaborazione completata. Risultati salvati in: {output_json_path}")