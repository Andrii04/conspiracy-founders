import pandas as pd
import json
import ast
import os


matching_dict_path = "data/interim/matching_dict_optimized.json"
users_folder_path = "data/raw/FOUNDERS_DATA/"
foundersFeatures_path = "data/interim/foundersFeatures_FINAL.csv"

# Percorso di output per il dataset unico dei founder
path_founders_scores_out = "data/interim/founders_conspiracy_scores.csv"

# Carica il vettore di somiglianza complottista esterno
conspiracy_vector_path = "data/external/conspiracy_svd_cossim_vector.csv"
df_conspiracy_vector = pd.read_csv(conspiracy_vector_path)
conspiracy_dict = dict(zip(df_conspiracy_vector["Unnamed: 0"], df_conspiracy_vector["similarity"]))

# Carica il dizionario dei match per ottenere la lista di tutti i founder
with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

# Carica il dataset centralizzato dei subreddit osservati per i founder
df_founders_features = pd.read_csv(foundersFeatures_path)


def give_score(subreddits):
    # Controllo di sicurezza se la lista è vuota
    if not subreddits:
        return 0.0

    total_conspiracy_load = 0.0
    total_contributions = 0

    for sub in subreddits:
        sub_name = sub.get("subreddit", "")
        count = sub.get("count", 0)

        # Cerca nel dizionario; se non esiste, restituisce 0.0
        similarity = conspiracy_dict.get(sub_name, 0.0)

        # Calcolo del carico complottista ponderato per i contributi
        total_conspiracy_load += (count * similarity)
        total_contributions += count

    # Evita l'errore "ZeroDivisionError" se i count sono a zero
    if total_contributions == 0:
        return 0.0

    # Media ponderata
    final_conspiracy_score = total_conspiracy_load / total_contributions

    return final_conspiracy_score


def founder_conspiracy_score():
    founders_scores = []

    for username in founders:
        
        # Filtra la riga corrispondente al founder corrente nel file osservato
        founder_row = df_founders_features[df_founders_features["username"] == username]
        
        if founder_row.empty:
            print(f"Dati di osservazione non trovati per {username}, salto.")
            continue
            
        raw_subreddits = founder_row["subreddits"].iloc[0]
        
        try:
            subreddits_list = ast.literal_eval(raw_subreddits) if isinstance(raw_subreddits, str) else raw_subreddits
        except (ValueError, SyntaxError):
            subreddits_list = []

        # Calcolo dello score sul profilo del founder
        conspiracy_score = give_score(subreddits_list)

        # Appende la riga alla lista globale del dataset unico
        final_row = {
            "author": username, 
            "subreddits": raw_subreddits, 
            "conspiracy_score": conspiracy_score
        }
        founders_scores.append(final_row)

    # Crea il DataFrame unico e lo salva in un singolo file CSV complessivo
    df_founders_scores = pd.DataFrame(founders_scores)
    os.makedirs(os.path.dirname(path_founders_scores_out), exist_ok=True)
    df_founders_scores.to_csv(path_founders_scores_out, index=False)
    
    print(f"\nFatto! Dataset unico dei founder creato in: {path_founders_scores_out}")


if __name__ == "__main__":
    founder_conspiracy_score()