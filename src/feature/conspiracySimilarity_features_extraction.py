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

output_json_path = "data/interim/user_conspiracy_similarity.json"

# FUNZIONI DI SUPPORTO

def safe_read_csv(filepath):
    """Legge un CSV in modo sicuro gestendo file inesistenti, vuoti (0 byte) o corrotti."""
    if not os.path.exists(filepath):
        return None
    if os.path.getsize(filepath) == 0:
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        return df
    except (pd.errors.EmptyDataError, Exception):
        return None

def is_valid_text(text):
    """Verifica che il testo sia valido ed esclude tag di sistema."""
    if pd.isna(text): return False
    s = str(text).strip()
    return bool(s) and s not in ["[deleted]", "[removed]"]

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

def get_mean_embedding_from_texts(texts_list, model, batch_size=32):
    """
    Estrae gli embedding per ogni singolo testo ed esegue il vettore medio del documento.
    Risolve il problema del troncamento a 512 token.
    """
    valid_texts = [str(t).strip() for t in texts_list if is_valid_text(t)]
    if not valid_texts:
        return None
    
    # Genera embedding per tutti i testi in batch (molto più veloce)
    embeddings = model.encode(valid_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
    
    # Ritorna il centroide / vettore medio
    return np.mean(embeddings, axis=0)

# ATTIVITÀ PRINCIPALE

if __name__ == "__main__":
    founders = load_founders_list()
    print(f"Caricati {len(founders)} founder da elaborare.")

    # 1. Caricamento del modello Sentence Transformer
    print("Caricamento del modello HuggingFace all-mpnet-base-v2...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # 2. Creazione dei vettori medi per i Subreddit del professore
    print("Costruzione dei vettori di riferimento per i subreddit conspiracy...")
    subreddit_texts = {}

    if os.path.exists(PROF_COMMENTS_PATH):
        with open(PROF_COMMENTS_PATH, "r") as f:
            prof_comments = json.load(f)
        for sub_name, comments_list in prof_comments.items():
            if sub_name not in subreddit_texts: subreddit_texts[sub_name] = []
            for c in comments_list:
                if isinstance(c, dict) and is_valid_text(c.get("body")):
                    subreddit_texts[sub_name].append(str(c.get("body")).strip())

    if os.path.exists(PROF_SUBMISSIONS_PATH):
        with open(PROF_SUBMISSIONS_PATH, "r") as f:
            prof_submissions = json.load(f)
        for sub_name, subs_list in prof_submissions.items():
            if sub_name not in subreddit_texts: subreddit_texts[sub_name] = []
            for s in subs_list:
                if isinstance(s, dict):
                    title = str(s.get("title", "")) if pd.notna(s.get("title")) else ""
                    body = str(s.get("selftext", "")) if pd.notna(s.get("selftext")) else ""
                    full_p = f"{title} {body}".strip()
                    if is_valid_text(full_p):
                        subreddit_texts[sub_name].append(full_p)

    # Generazione degli embedding medi per ogni subreddit
    subreddit_embeddings = {}
    for sub_name, texts in subreddit_texts.items():
        sub_vec = get_mean_embedding_from_texts(texts, model)
        if sub_vec is not None:
            subreddit_embeddings[sub_name] = sub_vec

    print(f"Generati vettori di benchmark per {len(subreddit_embeddings)} subreddit conspiracy.")

    # 3. Creazione documenti utenti e calcolo della somiglianza media
    print("\nInizio elaborazione e calcolo somiglianze per i founder...")
    conspiracy_similarity_results = {}

    for username in founders:
        folder = os.path.join(users_folder_path, f"{username}_data")
        posts_path = os.path.join(folder, f"{username}_posts.csv")
        comments_path = os.path.join(folder, f"{username}_comments.csv")

        user_text_fragments = []

        # Estrazione post dell'utente
        df_p = safe_read_csv(posts_path)
        if df_p is not None and "title" in df_p.columns and "selftext" in df_p.columns:
            for _, row in df_p.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else ""
                body = str(row['selftext']) if pd.notna(row['selftext']) else ""
                full_text = f"{title} {body}".strip()
                if is_valid_text(full_text):
                    user_text_fragments.append(full_text)

        # Estrazione commenti dell'utente
        df_c = safe_read_csv(comments_path)
        if df_c is not None and "body" in df_c.columns:
            for _, row in df_c.iterrows():
                body = str(row['body']) if pd.notna(row['body']) else ""
                if is_valid_text(body):
                    user_text_fragments.append(body)

        # Calcolo del vettore medio dell'utente
        user_vector = get_mean_embedding_from_texts(user_text_fragments, model)

        if user_vector is None:
            print(f"Mancanza di testo per l'utente {username}, score impostato a 0.0")
            mean_similarity = 0.0
        else:
            similarities = []
            for sub_name, sub_vector in subreddit_embeddings.items():
                sim = calculate_cosine_similarity(user_vector, sub_vector)
                similarities.append(sim)

            mean_similarity = float(np.mean(similarities)) if similarities else 0.0

        conspiracy_similarity_results[username] = mean_similarity

        # Salvataggio del CSV singolo per l'utente (per create_final_dataset.py)
        user_csv_out = os.path.join(folder, f"{username}_conspiracySimilarity_FEATURES.csv")
        df_user_feat = pd.DataFrame([{
            "author": username,
            "avg_conspiracy_similarity": mean_similarity
        }])
        df_user_feat.to_csv(user_csv_out, index=False)

        print(f"Utente {username} elaborato. Cosine Similarity Media: {mean_similarity:.4f}")

    # 4. Salvataggio finale del dizionario globale in formato JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(conspiracy_similarity_results, f, indent=4)

    print(f"\nElaborazione completata. Risultati salvati in: {output_json_path}")