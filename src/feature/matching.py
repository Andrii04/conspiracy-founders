import numpy as np
import pandas as pd
import ast
from gensim.models import KeyedVectors
from tqdm import tqdm
from dimen_generation import load_embedding
import json

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
    bins = [0, 50, 200, 500, 1000, 2500, np.inf]
    bin_labels = ["0-50", "50-200", "200-500", "500-1000", "1000-2500", "2500+"]

    df["num_contrib_bin"] = pd.cut(df["num contributions"], bins=bins, labels=bin_labels, include_lowest=True)
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

def match_users(df_anchors, df_candidates):
    # Allineamento con i documenti disponibili
    anchors = [u for u in df_anchors["username"] if u in user_documents]
    candidates = [u for u in df_candidates["username"] if u in user_documents]

    # INIZIALIZZAZIONE DIZIONARIO DI DIZIONARI (Sparse Representation)
    matching_dict = {}

    print("Avvio matching con filtraggio per bin contributi e anno prima attività")

    # Ciclo diretto sugli elementi (senza indici i, j, molto più pythonic)
    for user_a in tqdm(anchors):
        # Inizializziamo il dizionario interno per l'anchor corrente
        matching_dict[user_a] = {}
        info_a = user_info[user_a]

        for user_b in candidates:
            # CASO 1: Identità (Distanza zero con se stessi)
            if user_a == user_b:
                matching_dict[user_a][user_b] = 0.0
                continue  # Salta direttamente al prossimo candidato, ottimizzando i tempi

            info_b = user_info[user_b]

            # Controllo dei vincoli del Blocco
            same_contrib_bin = (info_a["num_contrib_bin"] == info_b["num_contrib_bin"])
            same_year = (info_a["first_activity_year"] == info_b["first_activity_year"])

            # CASO 2: Superamento del Blocco (Calcolo effettivo della WMD)
            if same_contrib_bin and same_year:
                distance = kv_model.wmdistance(user_documents[user_a], user_documents[user_b])
                # Salviamo nel dizionario interno solo se è un match valido
                matching_dict[user_a][user_b] = distance
    
    print("Matching completato con successo.")
    return matching_dict


if __name__ == "__main__":
    
    matching_dict = match_users(df_anchors, df_candidates)

    #salvataggio
    path_output_json = "../data/raw/matching_dict_raw.json"
    with open(path_output_json, "w") as f:
        json.dump(matching_dict, f, indent=4)


    print("\nCONTEGGIO CANDIDATI VALIDI PER ANCHOR")
    
    # Mostriamo i primi 20 risultati come diagnostica per non intasare l'output
    for anchor in list(matching_dict.keys())[:20]:
        num_matches = len(matching_dict[anchor])
        print(f"Anchor: {anchor:<25} | Candidati idonei nel bin: {num_matches}")