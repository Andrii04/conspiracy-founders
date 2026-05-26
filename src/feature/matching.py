import numpy as np
import pandas as pd
import ast
from gensim.models import KeyedVectors
from tqdm import tqdm


df = pd.read_csv("/home/andri/conspiracy-founders/data/interim/realFounders_features_clean.csv")




def load_embedding(embedding_folder = '../../data/external/embedding'):
    vectors_path = f"{embedding_folder}/vectors.tsv"
    metadata_path = f"{embedding_folder}/metadata.tsv"

    meta = pd.read_csv(metadata_path, sep='\t', header=None)
    meta.columns = meta.iloc[0]
    meta = meta.reindex(meta.index.drop(0))
    meta.set_index(meta.columns[0], inplace=True)

    vectors = pd.read_csv(vectors_path, sep='\t', header=None)
    vectors.set_index(meta.index, inplace=True)

    # Normalizzazione per rendere vettori compatibili e confrontabili
    vectors = vectors.divide(np.linalg.norm(vectors.values, axis=1), axis="rows")

    return vectors, meta

vectors, metadata = load_embedding()
print(f"caricati {len(vectors)} subreddit con vettory a {vectors.shape[1]} dimensioni.\n")




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
    for subreddit, contributions in subreddits_dict.items():
        if subreddit in kv_model:
            document.extend([subreddit] * contributions)
        else:
            print(f"Dumped {subreddit} because not in embeddings")
    
    return document




# dizionario con dentro i documenti finali degli utenti pronti per la WMD
user_documents = {}


for index, row in df.iterrows():
    username = row["username"]
    user_subreddits = ast.literal_eval(row["subreddits"])
    user_document = turn_user_into_document(user_subreddits, kv_model)

    if len(user_document) > 0:
        user_documents[username] = user_document




# Creazione bin per num contributi
def contributions_bins(dataset=df):
    bins = [0, 200, 500, 1000, 5000, np.inf]
    bin_labels = ["0-200", "200-500", "500-1000", "1000-5000", "5000+"]

    df["num_contrib_bin"] = pd.cut(df["num contributions"], bins=bins, labels=bin_labels)
    df["first_activity_year"] = pd.to_datetime(df["active since"], unit='s').dt.year


contributions_bins()
user_info = df.set_index("username")[["num_contrib_bin", "first_activity_year"]].to_dict("index")



# FUNZIONE PROVVISORIA (solo per vedere se funziona matching, test matching FRA FOUNDER)
def match_users():

    # matrice finale con distanze fra utenti
    usernames = list(user_documents.keys())
    n_users = len(usernames)

    # Inizializziamo matrice con NaN (che significa nessun match calcolato)
    matrix = np.full((n_users, n_users), np.nan)

    # distanza di un utene con se stesso è 0
    np.fill_diagonal(matrix, 0)

    print("Avvio matching con filtraggio per bin contributi e anno prima attività")

    for i in tqdm(range(n_users)):
        user_a = usernames[i]

        for j in range(i+1, n_users):
            user_b = usernames[j]

            info_a = user_info[user_a]
            info_b = user_info[user_b]

            same_contrib_bin = (info_a["num_contrib_bin"] == info_b["num_contrib_bin"])
            same_year = (info_a["first_activity_year"] == info_b["first_activity_year"])

            if same_contrib_bin and same_year:
                # Calcolo WMD subreddit
                distance = kv_model.wmdistance(user_documents[user_a], user_documents[user_b])
                matrix[i, j] = distance
                matrix[j, i] = distance
    
    # trasformiamo in DataFrame
    df_matching = pd.DataFrame(matrix, index=usernames, columns=usernames)
    print("Matrix complete")

    return df_matching




df_matching_test = match_users()