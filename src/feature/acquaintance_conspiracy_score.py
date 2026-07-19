import pandas as pd
import json
import ast  
from config import OBSERVATION_WINDOW_SECONDS
from config import CC_THRESHOLD



matching_dict_path = "data/interim/matching_dict_optimized.json"
users_folder_path = "data/raw/FOUNDERS_DATA/"

conspiracy_vector_path = "data/external/conspiracy_svd_cossim_vector.csv"
df_conspiracy_vector = pd.read_csv(conspiracy_vector_path)

conspiracy_dict = dict(zip(df_conspiracy_vector["Unnamed: 0"], df_conspiracy_vector["similarity"]))

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
# get all founders usernames
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates



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

    # media ponderata
    final_conspiracy_score = total_conspiracy_load / total_contributions

    return final_conspiracy_score



def acquaintance_conspiracy_score():

    for username in founders:
        
        acquaintances_scores = []

        # load acquaintances
        acquaintances_path = users_folder_path + username + "_data/" + username + "_acquaintancesSubreddits.csv"
        
        # Controllo se il file esiste (previene crash se un founder non ha acquaintances)
        import os
        if not os.path.exists(acquaintances_path):
            print(f"File non trovato per {username}, salto.")
            continue
            
        df_acquaintances_subreddits = pd.read_csv(acquaintances_path)

        for row in df_acquaintances_subreddits.itertuples():
            author = row.author
            raw_subreddits = row.subreddits
            
            try:
                subreddits_list = ast.literal_eval(raw_subreddits) if isinstance(raw_subreddits, str) else raw_subreddits
            except (ValueError, SyntaxError):
                subreddits_list = []

            # Calcolo dello score
            conspiracy_score = give_score(subreddits_list)

            final_row = {"author" : author, "subreddits" : raw_subreddits, "conspiracy_score" : conspiracy_score}
            acquaintances_scores.append(final_row)

        path_acquaintances_scores_out = users_folder_path + username + "_data/" + username + "_acquaintancesScores.csv"
        df_acquaintances_scores = pd.DataFrame(acquaintances_scores)

        df_acquaintances_scores.to_csv(path_acquaintances_scores_out, index=False)
        print(f"Completato {username}")



if __name__ == "__main__":
    acquaintance_conspiracy_score()