import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import os
import json

# Inizializzazione degli Strumenti NLP
print("Caricamento modelli NLP in corso...")
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
empath = Empath()

# Scegli qui le categorie Empath rilevanti per la tua tesi sui complotti
TARGET_EMPATH_CATEGORIES = ["fear", "violence", "government", "power", "deception"]

users_folder_path = "data/raw/FOUNDERS_DATA/"
path_founders_psy_out = "data/interim/founders_psy_features.csv"

matching_dict_path = "data/interim/matching_dict_optimized.json"

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
# get all founders usernames
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

all_founders_profiles = []

# Funzioni di Supporto per l'estrazione riga per riga
def get_vader_compound(text):
    if pd.isna(text) or text == "": return 0.0
    return vader.polarity_scores(str(text))["compound"]

def get_empath_scores(text):
    if pd.isna(text) or text == "": 
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    scores = empath.analyze(str(text), normalize=True)
    if not scores: # Se Empath non riconosce parole valide
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    return {cat: scores.get(cat, 0.0) for cat in TARGET_EMPATH_CATEGORIES}

def get_spacy_features(text):
    if pd.isna(text) or text == "": 
        return 0, 0, 0.0
    doc = nlp(str(text))
    
    # NER: Conteggio enti/persone
    ner_count = len([ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]])
    
    # POS: Calcolo percentuale pronomi rispetto alla lunghezza del testo
    total_tokens = len(doc)
    if total_tokens == 0: return ner_count, 0, 0.0
    pronoun_count = len([token for token in doc if token.pos_ == "PRON"])
    pronoun_ratio = pronoun_count / total_tokens
    
    return ner_count, pronoun_count, pronoun_ratio

# Main Loop: Elaborazione per Founder
for username in founders:
    print(f"Elaborazione features psicologiche per: {username}")
    
    founder_folder = os.path.join(users_folder_path, username + "_data")
    comments_path = os.path.join(founder_folder, username + "_comments.csv")
    posts_path = os.path.join(founder_folder, username + "_posts.csv")
    
    all_texts = []
    
    # Caricamento e fusione dei testi (Commenti)
    if os.path.exists(comments_path):
        df_c = pd.read_csv(comments_path)
        if "body" in df_c.columns:
            all_texts.extend(df_c["body"].dropna().astype(str).tolist())
            
    # Caricamento e fusione dei testi (Post)
    if os.path.exists(posts_path):
        df_p = pd.read_csv(posts_path)
        # Unisco title e selftext (gestendo i NaN con stringhe vuote)
        if "title" in df_p.columns and "selftext" in df_p.columns:
            titles = df_p["title"].fillna("").astype(str)
            bodies = df_p["selftext"].fillna("").astype(str)
            combined_posts = (titles + " " + bodies).tolist()
            all_texts.extend(combined_posts)
            
    if not all_texts:
        print(f"   Nessun testo trovato per {username}, salto.")
        continue
        
    df_texts = pd.DataFrame({"text": all_texts})
    
    # Estrazione features per ogni singola riga
    df_texts["vader"] = df_texts["text"].apply(get_vader_compound)
    
    empath_results = df_texts["text"].apply(get_empath_scores)
    for cat in TARGET_EMPATH_CATEGORIES:
        df_texts[f"empath_{cat}"] = empath_results.apply(lambda x: x[cat])
        
    spacy_results = df_texts["text"].apply(get_spacy_features)
    df_texts["ner_total"] = spacy_results.apply(lambda x: x[0])
    df_texts["pronoun_ratio"] = spacy_results.apply(lambda x: x[2])
    
    # Aggregazione finale del profilo Founder (1 riga)
    founder_profile = {
        "author": username,
        "total_contributions": len(df_texts),
        "avg_sentiment": df_texts["vader"].mean(),
        "total_ner_mentions": df_texts["ner_total"].sum(),
        "avg_pronoun_ratio": df_texts["pronoun_ratio"].mean()
    }
    
    # Aggiungo dinamicamente le medie delle categorie Empath
    for cat in TARGET_EMPATH_CATEGORIES:
        founder_profile[f"avg_empath_{cat}"] = df_texts[f"empath_{cat}"].mean()
        
    all_founders_profiles.append(founder_profile)

    # Salvataggio nel file CSV specifico del founder
    founder_psy_path = os.path.join(founder_folder, f"{username}_psy.csv")
    df_single_founder = pd.DataFrame([founder_profile])
    df_single_founder.to_csv(founder_psy_path, index=False)

# Salvataggio in un unico file CSV globale
if len(all_founders_profiles) == len(founders):
    print("\nTutti i founder sono stati elaborati.")
else:
    print(f"\nElaborati {len(all_founders_profiles)} founder su {len(founders)} totali.")

if all_founders_profiles:
    df_final = pd.DataFrame(all_founders_profiles)
    df_final.to_csv(path_founders_psy_out, index=False)
    print(f"Elaborazione completata. Profili aggregati salvati in: {path_founders_psy_out}")
else:
    print("\nNessun dato estratto.")

