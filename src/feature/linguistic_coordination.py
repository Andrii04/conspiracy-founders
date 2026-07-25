import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import os
import json

CC_THRESHOLD = 0.15

print("Loading NLP models for linguistic coordination...")
vader = SentimentIntensityAnalyzer()
empath = Empath()

# Standalone categories for NLP feature vectors
TARGET_EMPATH_CATEGORIES = [
    "fear", "violence", "power", "deception", "anger", "sadness", "nervousness",
    "dominant_personality", "pride", "optimism", "shame", "irritability", "trust",
    "politeness", "sympathy", "cheerfulness", "negative_emotion", "aggression",
    "hate", "swearing_terms", "envy", "exasperation", "contentment", "positive_emotion"
]

# Subsets for positive and negative linguistic coordination components
POSITIVE_CATEGORIES = ["optimism", "trust", "politeness", "sympathy", "cheerfulness", "contentment", "positive_emotion"]
NEGATIVE_CATEGORIES = ["fear", "violence", "anger", "sadness", "irritability", "negative_emotion", "aggression", "hate", "swearing_terms", "exasperation", "nervousness"]

users_folder_path = "data/raw/FOUNDERS_DATA/"
matching_dict_path = "data/interim/matching_dict_optimized.json"

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
anchors = list(matching_dict.keys())
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

def safe_read_csv(filepath):
    """Legge un file CSV in modo sicuro gestendo file inesistenti, vuoti o corrotti."""
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

def clean_id(val):
    """Converte in stringa pulita ed elimina spazi vuoti."""
    if pd.isna(val): return ""
    return str(val).strip()

def is_valid_text(text):
    """Verifica se il testo è valido per l'analisi NLP."""
    s = str(text).strip()
    return bool(s) and s not in ["[deleted]", "[removed]"]

def get_nlp_features(text):
    """Estrae le feature VADER ed Empath per un testo."""
    if not is_valid_text(text):
        features = {"vader": 0.0}
        for cat in TARGET_EMPATH_CATEGORIES:
            features[cat] = 0.0
        return features
        
    features = {}
    features["vader"] = vader.polarity_scores(str(text))["compound"]
    
    empath_scores = empath.analyze(str(text), normalize=True)
    for cat in TARGET_EMPATH_CATEGORIES:
        features[cat] = empath_scores.get(cat, 0.0) if empath_scores else 0.0
        
    return features

def calculate_coordination(vec1, vec2):
    """Calcola la Cosine Similarity tra due vettori."""
    if np.all(vec1 == 0) or np.all(vec2 == 0): 
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: 
        return 0.0
    return dot_product / (norm1 * norm2)

for username in founders:
    print(f"Processing linguistic coordination for: {username}")
    
    founder_folder = os.path.join(users_folder_path, username + "_data")
    comments_path = os.path.join(founder_folder, username + "_comments.csv")
    comments_recv_path = os.path.join(founder_folder, username + "_commentsReceived.csv")
    posts_commented_path = os.path.join(founder_folder, username + "_postsCommented.csv")
    acq_scores_path = os.path.join(founder_folder, username + "_acquaintancesScores.csv")
    
    out_path = os.path.join(founder_folder, f"{username}_linguisticCoordination.csv")
    
    # Caricamento sicuro dei commenti del founder
    df_comments = safe_read_csv(comments_path)
    if df_comments is None:
        print(f"No valid/non-empty reply comments found for {username}; skipping.")
        continue
        
    # Map acquaintance CC status
    cc_map = {}
    df_scores = safe_read_csv(acq_scores_path)
    if df_scores is not None and "author" in df_scores.columns and "conspiracy_score" in df_scores.columns:
        for row in df_scores.itertuples():
            cc_map[clean_id(row.author)] = float(row.conspiracy_score) > CC_THRESHOLD
                
    # Map parent fullname (t1_... o t3_...) to replied text and author
    parents_dict = {}
    
    # 1. Carica commenti ricevuti
    df_recv = safe_read_csv(comments_recv_path)
    if df_recv is not None:
        for row in df_recv.itertuples():
            fullname = clean_id(getattr(row, 'name', ''))
            author = clean_id(getattr(row, 'author', ''))
            body = getattr(row, 'body', '')
            if fullname and is_valid_text(body):
                parents_dict[fullname] = {"author": author, "text": str(body)}
            
    # 2. Carica post di conoscenti che il founder ha commentato
    df_posts = safe_read_csv(posts_commented_path)
    if df_posts is not None:
        for row in df_posts.itertuples():
            fullname = clean_id(getattr(row, 'name', ''))
            author = clean_id(getattr(row, 'author', ''))
            title = str(row.title) if pd.notna(getattr(row, 'title', None)) else ""
            body = str(row.selftext) if pd.notna(getattr(row, 'selftext', None)) else ""
            combined_text = f"{title} {body}".strip()
            
            if fullname and is_valid_text(combined_text):
                parents_dict[fullname] = {"author": author, "text": combined_text}
            
    interactions_data = []
    
    # Processa i commenti scritti dal Founder
    for row in df_comments.itertuples():
        parent_id = clean_id(getattr(row, 'parent_id', ''))
        founder_text = getattr(row, 'body', '')
        
        # Mantiene solo interazioni di risposta tracciate e con testo valido
        if parent_id in parents_dict and is_valid_text(founder_text):
            acq_data = parents_dict[parent_id]
            acq_author = acq_data["author"]
            
            # FILTRO AUTO-RISPOSTE: Esclude i casi in cui il founder risponde a se stesso
            if acq_author.lower() == username.lower():
                continue
                
            acq_text = acq_data["text"]
            
            f_feats = get_nlp_features(founder_text)
            a_feats = get_nlp_features(acq_text)
            
            # 1. Vettori globali
            vec_f = np.array([f_feats["vader"]] + [f_feats[c] for c in TARGET_EMPATH_CATEGORIES])
            vec_a = np.array([a_feats["vader"]] + [a_feats[c] for c in TARGET_EMPATH_CATEGORIES])
            coord_score = calculate_coordination(vec_f, vec_a)
            
            # 2. Vettori pos
            vec_f_pos = np.array([f_feats[c] for c in POSITIVE_CATEGORIES])
            vec_a_pos = np.array([a_feats[c] for c in POSITIVE_CATEGORIES])
            pos_coord_score = calculate_coordination(vec_f_pos, vec_a_pos)
            
            # 3. Vettori neg
            vec_f_neg = np.array([f_feats[c] for c in NEGATIVE_CATEGORIES])
            vec_a_neg = np.array([a_feats[c] for c in NEGATIVE_CATEGORIES])
            neg_coord_score = calculate_coordination(vec_f_neg, vec_a_neg)
            
            is_cc = cc_map.get(acq_author, False)
            
            comment_reddit_id = clean_id(getattr(row, 'name', getattr(row, 'id', '')))
            
            row_data = {
                "interaction_id": comment_reddit_id,
                "parent_id": parent_id,
                "link_id": clean_id(getattr(row, 'link_id', '')),
                "founder": username,
                "acquaintance": acq_author,
                "is_cc": is_cc,
                "coordination_score": coord_score,
                "positive_coordination_score": pos_coord_score,
                "negative_coordination_score": neg_coord_score,
                "founder_vader": f_feats["vader"],
                "acq_vader": a_feats["vader"]
            }
            
            for cat in TARGET_EMPATH_CATEGORIES:
                row_data[f"founder_empath_{cat}"] = f_feats[cat]
                row_data[f"acq_empath_{cat}"] = a_feats[cat]
                
            interactions_data.append(row_data)
            
    if interactions_data:
        df_coord = pd.DataFrame(interactions_data)
        df_coord.to_csv(out_path, index=False)
        print(f"Saved dataset for {username} ({len(interactions_data)} matched interactions).")
    else:
        print(f"No matchable interactions for {username}.")
        
print("Global processing completed.")