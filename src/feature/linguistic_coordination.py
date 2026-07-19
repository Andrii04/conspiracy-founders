import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import os
import json

print("Loading NLP models for linguistic coordination...")
vader = SentimentIntensityAnalyzer()
empath = Empath()

# All standalone categories requested from the psych features script
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

# Extract NLP features for one text
def get_nlp_features(text):
    if pd.isna(text) or str(text).strip() == "":
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

# Compute linguistic coordination (cosine similarity)
def calculate_coordination(vec1, vec2):
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
    
    if not os.path.exists(comments_path):
        print(f"No reply comments found for {username}; skipping.")
        continue
        
    # Map acquaintance CC status
    cc_map = {}
    if os.path.exists(acq_scores_path):
        df_scores = pd.read_csv(acq_scores_path)
        if "author" in df_scores.columns and "conspiracy_score" in df_scores.columns:
            for row in df_scores.itertuples():
                # Mark as CC when score > 0.5
                cc_map[str(row.author)] = float(row.conspiracy_score) > 0.5
                
    # Map parent id to replied text and author
    parents_dict = {}
    
    if os.path.exists(comments_recv_path):
        df_recv = pd.read_csv(comments_recv_path)
        for row in df_recv.itertuples():
            parents_dict[str(row.name)] = {"author": str(row.author), "text": str(row.body)}
            
    if os.path.exists(posts_commented_path):
        df_posts = pd.read_csv(posts_commented_path)
        for row in df_posts.itertuples():
            title = str(row.title) if pd.notna(row.title) else ""
            body = str(row.selftext) if pd.notna(row.selftext) else ""
            parents_dict[str(row.name)] = {"author": str(row.author), "text": title + " " + body}
            
    df_comments = pd.read_csv(comments_path)
    interactions_data = []
    
    # Iterate founder comments
    for row in df_comments.itertuples():
        parent_id = str(row.parent_id)
        
        # Keep only tracked reply interactions
        if parent_id in parents_dict:
            acq_data = parents_dict[parent_id]
            acq_author = acq_data["author"]
            
            founder_text = str(row.body)
            acq_text = acq_data["text"]
            
            f_feats = get_nlp_features(founder_text)
            a_feats = get_nlp_features(acq_text)
            
            # 1. Build global feature vectors for total coordination score
            vec_f = np.array([f_feats["vader"]] + [f_feats[c] for c in TARGET_EMPATH_CATEGORIES])
            vec_a = np.array([a_feats["vader"]] + [a_feats[c] for c in TARGET_EMPATH_CATEGORIES])
            coord_score = calculate_coordination(vec_f, vec_a)
            
            # 2. Build sub-vectors for positive coordination score
            vec_f_pos = np.array([f_feats[c] for c in POSITIVE_CATEGORIES])
            vec_a_pos = np.array([a_feats[c] for c in POSITIVE_CATEGORIES])
            pos_coord_score = calculate_coordination(vec_f_pos, vec_a_pos)
            
            # 3. Build sub-vectors for negative coordination score
            vec_f_neg = np.array([f_feats[c] for c in NEGATIVE_CATEGORIES])
            vec_a_neg = np.array([a_feats[c] for c in NEGATIVE_CATEGORIES])
            neg_coord_score = calculate_coordination(vec_f_neg, vec_a_neg)
            
            is_cc = cc_map.get(acq_author, False)
            
            # Build output row
            row_data = {
                "interaction_id": row.name,
                "parent_id": parent_id,
                "link_id": row.link_id,
                "founder": username,
                "acquaintance": acq_author,
                "is_cc": is_cc,
                "coordination_score": coord_score,
                "positive_coordination_score": pos_coord_score,
                "negative_coordination_score": neg_coord_score,
                "founder_vader": f_feats["vader"],
                "acq_vader": a_feats["vader"]
            }
            
            # Add Empath feature pairs
            for cat in TARGET_EMPATH_CATEGORIES:
                row_data[f"founder_empath_{cat}"] = f_feats[cat]
                row_data[f"acq_empath_{cat}"] = a_feats[cat]
                
            interactions_data.append(row_data)
            
    # Save dataset
    if interactions_data:
        df_coord = pd.DataFrame(interactions_data)
        df_coord.to_csv(out_path, index=False)
        print(f"Saved dataset for {username} ({len(interactions_data)} matched interactions).")
    else:
        print(f"No matchable interactions for {username}.")
        
print("Global processing completed.")