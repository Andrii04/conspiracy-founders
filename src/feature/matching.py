import numpy as np
import pandas as pd
import ast
from gensim.models import KeyedVectors
from tqdm import tqdm
from dimen_generation import load_embedding

PATH_ANCHORS = "data/interim/anchors_test.csv"
PATH_CANDIDATES = "data/interim/candidates_test.csv"

df_anchors = pd.read_csv(PATH_ANCHORS, index_col=0)
df_candidates = pd.read_csv(PATH_CANDIDATES, index_col=0)

vectors, metadata = load_embedding()
print(f"caricati {len(vectors)} subreddit con vettori a {vectors.shape[1]} dimensioni.\n")




# Conversione in formato Gensim (inserire vettori subreddit in KeyedVectors come se fossero parole)
# Inizializzazione
vector_size = vectors.shape[1]
kv_model = KeyedVectors(vector_size=vector_size)

# Estraiamo nomi subreddit e matrice numerica
subreddit_names = vectors.index.tolist()
vector_matrix = vectors.values.astype("float32")

# Carichiamo dentro il modello Gensim
kv_model.add_vectors(subreddit_names, vector_matrix)

print("Vettori convertiti in formato Gensim\n")




# Preparazione Dati Utenti (documenti)
def turn_user_into_document(subreddits_dict, kv_model):
    # prende subreddit utente, filtra i subreddit out of vocabulary,
    # e srotola dizionario in una lista di stringhe pesata
    document = []

    for item in subreddits_dict:
        subreddit = item["subreddit"]
        contributions = item["count"]

        if subreddit in kv_model:
            document.extend([subreddit] * contributions)
        else:  
            print(f"Dumped {subreddit} because not in embeddings")

    return document



# dizionario con dentro i documenti finali degli utenti pronti per la WMD
user_documents = {}
user_info = {}

# Creazione bin per num contributi
def contributions_bins(df):
    bins = [0, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12500, 15000, 17500, 20000, 40000, 60000, 80000, 100000, np.inf]
    bin_labels = ["0-100", "100-250", "250-500", "500-750", "750-1000", "1000-2000", "2000-3000", "3000-4000", "4000-5000", "5000-6000", "6000-7000", "7000-8000", "8000-9000", "9000-10000", "10000-12500", "12500-15000", "15000-17500", "17500-20000", "20000-40000", "40000-60000", "60000-80000", "80000-100000", "100000+"]

    df["num_contrib_bin"] = pd.cut(df["num contributions"], bins=bins, labels=bin_labels)
    df["first_activity_year"] = pd.to_datetime(df["active since"], unit='s').dt.year

def fill_documents(df):

    contributions_bins(df)

    for index, row in df.iterrows():
        username = row["username"]
        user_subreddits = ast.literal_eval(row["subreddits"])
        user_document = turn_user_into_document(user_subreddits, kv_model)

        if len(user_document) > 0:
            user_documents[username] = user_document

    info_dict = df.set_index("username")[["num_contrib_bin", "first_activity_year"]].to_dict("index")
    user_info.update(info_dict)


fill_documents(df_anchors)
fill_documents(df_candidates)


# FUNZIONE PROVVISORIA (solo per vedere se funziona matching, test matching FRA FOUNDER) output: matrice
# dove matrice[i][j] = distanza fra user i e user j
#anchor: founders, candidates: user boh chissà quanti
def match_users(anchors, candidates):

    anchors = [u for u in df_anchors["username"] if u in user_documents]
    candidates = [u for u in df_candidates["username"] if u in user_documents]
    # dividere set founder in 2 per test (modificare sript)

    n_anchors = len(anchors)
    n_candidates = len(candidates)

    # Inizializziamo matrice con NaN (che significa nessun match calcolato)
    matrix = np.full((n_anchors, n_candidates), np.nan)

    # distanza di un utene con se stesso è 0

    print("Avvio matching con filtraggio per bin contributi e anno prima attività")

    for i in tqdm(range(n_anchors)):
        user_a = anchors[i]
        info_a = user_info[user_a]

        for j in range(n_candidates):
            user_b = candidates[j]

            if user_a == user_b:
                matrix[i][j] = 0

            info_b = user_info[user_b]

            same_contrib_bin = (info_a["num_contrib_bin"] == info_b["num_contrib_bin"])
            same_year = (info_a["first_activity_year"] == info_b["first_activity_year"])

            if same_contrib_bin and same_year:
                # Calcolo WMD subreddit
                distance = kv_model.wmdistance(user_documents[user_a], user_documents[user_b])
                matrix[i, j] = distance
    
    # trasformiamo in DataFrame
    df_matching = pd.DataFrame(matrix, index=anchors, columns=candidates)
    print("Matrix complete")

    return df_matching



if __name__ == "__main__":
    df_matching = match_users(df_anchors, df_candidates)
    # df_matching.to_csv("data/raw/matching_matrix_raw.csv")

    # 
    print(df_matching.notna().sum(axis=1))