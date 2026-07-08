import pandas as pd
import spacy
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import os
import json

# Initialize NLP tools
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
empath = Empath()

# Target Empath categories
TARGET_EMPATH_CATEGORIES = ["fear", "violence", "government", "power", "deception", "anger", "sadness", "nervousness"]

# CC threshold
CC_THRESHOLD = 0.02

users_folder_path = "data/raw/FOUNDERS_DATA/"
path_founders_psy_out = "data/interim/founders_psy_features.csv"

matching_dict_path = "data/interim/matching_dict_optimized.json"

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
# Collect founder usernames
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

all_founders_profiles = []

# Row-level helpers
def get_vader_compound(text):
    if pd.isna(text) or text == "": return 0.0
    return vader.polarity_scores(str(text))["compound"]

def get_empath_scores(text):
    if pd.isna(text) or text == "": 
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    scores = empath.analyze(str(text), normalize=True)
    if not scores:  # No valid Empath tokens found
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    return {cat: scores.get(cat, 0.0) for cat in TARGET_EMPATH_CATEGORIES}

def get_spacy_features(text):
    if pd.isna(text) or text == "": 
        return 0, 0, 0.0
    doc = nlp(str(text))
    
    # Count PERSON and ORG entities
    ner_count = len([ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]])
    
    # Pronoun ratio over tokens
    total_tokens = len(doc)
    if total_tokens == 0: return ner_count, 0, 0.0
    pronoun_count = len([token for token in doc if token.pos_ == "PRON"])
    pronoun_ratio = pronoun_count / total_tokens
    
    return ner_count, pronoun_count, pronoun_ratio

# Safe aggregations for empty slices
def safe_mean(series):
    return series.mean() if not series.dropna().empty else 0.0

def safe_sum(series):
    return series.sum() if not series.dropna().empty else 0

# Main loop per founder
for username in founders:
    print(f"Processing psychological features for: {username}")
    
    founder_folder = os.path.join(users_folder_path, username + "_data")
    comments_path = os.path.join(founder_folder, username + "_comments.csv")
    posts_path = os.path.join(founder_folder, username + "_posts.csv")
    
    # Files for CC context and linguistic coordination
    acq_path = os.path.join(founder_folder, username + "_acquaintancesScores.csv")
    recv_path = os.path.join(founder_folder, username + "_commentsReceived.csv")
    posts_comm_path = os.path.join(founder_folder, username + "_postsCommented.csv")
    coord_path = os.path.join(founder_folder, username + "_linguisticCoordination.csv")
    
    # Build CC map for acquaintances
    cc_map = {}
    if os.path.exists(acq_path):
        df_acq = pd.read_csv(acq_path)
        if 'author' in df_acq.columns and 'conspiracy_score' in df_acq.columns:
            for _, row in df_acq.iterrows():
                cc_map[str(row['author'])] = float(row['conspiracy_score']) > CC_THRESHOLD
                
    # Build target-author map for replies
    target_author_map = {}
    if os.path.exists(recv_path):
        df_recv = pd.read_csv(recv_path)
        if 'name' in df_recv.columns and 'author' in df_recv.columns:
            for _, row in df_recv.iterrows():
                target_author_map[str(row['name'])] = str(row['author'])
    if os.path.exists(posts_comm_path):
        df_p_comm = pd.read_csv(posts_comm_path)
        if 'name' in df_p_comm.columns and 'author' in df_p_comm.columns:
            for _, row in df_p_comm.iterrows():
                target_author_map[str(row['name'])] = str(row['author'])

    texts_data = []
    
    # Load and merge comment texts with CC tag
    if os.path.exists(comments_path):
        df_c = pd.read_csv(comments_path)
        if 'body' in df_c.columns and 'parent_id' in df_c.columns:
            for _, row in df_c.iterrows():
                body = str(row['body']) if pd.notna(row['body']) else ""
                if body.strip() == "": continue
                
                parent_id = str(row['parent_id'])
                target_author = target_author_map.get(parent_id)
                
                # Default to non-CC when target is missing
                is_cc = cc_map.get(target_author, False) if target_author else False
                
                texts_data.append({"text": body, "is_cc": is_cc})
                
    # Load and merge post texts
    if os.path.exists(posts_path):
        df_p = pd.read_csv(posts_path)
        if "title" in df_p.columns and "selftext" in df_p.columns:
            for _, row in df_p.iterrows():
                title = str(row['title']) if pd.notna(row['title']) else ""
                body = str(row['selftext']) if pd.notna(row['selftext']) else ""
                text = f"{title} {body}".strip()
                if text == "": continue
                
                # Posts are not replies, so keep is_cc as NA
                texts_data.append({"text": text, "is_cc": pd.NA})
                
    if not texts_data:
        print(f"No valid text found for {username}; skipping.")
        continue
        
    df_texts = pd.DataFrame(texts_data)
    
    # Extract row-level features
    df_texts["vader"] = df_texts["text"].apply(get_vader_compound)
    
    empath_results = df_texts["text"].apply(get_empath_scores)
    for cat in TARGET_EMPATH_CATEGORIES:
        df_texts[f"empath_{cat}"] = empath_results.apply(lambda x: x[cat])
        
    spacy_results = df_texts["text"].apply(get_spacy_features)
    df_texts["ner_total"] = spacy_results.apply(lambda x: x[0])
    df_texts["pronoun_ratio"] = spacy_results.apply(lambda x: x[2])
    
    # Masks for CC and non-CC subsets
    mask_cc = df_texts['is_cc'] == True
    mask_non_cc = df_texts['is_cc'] == False
    
    # Compute linguistic coordination
    avg_coord = 0.0
    avg_coord_cc = 0.0
    avg_coord_non_cc = 0.0
    
    if os.path.exists(coord_path):
        df_coord = pd.read_csv(coord_path)
        if 'coordination_score' in df_coord.columns and 'is_cc' in df_coord.columns:
            avg_coord = safe_mean(df_coord['coordination_score'])
            
            mask_coord_cc = df_coord['is_cc'] == True
            mask_coord_non_cc = df_coord['is_cc'] == False
            
            avg_coord_cc = safe_mean(df_coord[mask_coord_cc]['coordination_score'])
            avg_coord_non_cc = safe_mean(df_coord[mask_coord_non_cc]['coordination_score'])
    
    # Aggregate founder profile
    founder_profile = {
        "author": username,
        "total_contributions": len(df_texts),
        
        # Sentiment
        "avg_sentiment": safe_mean(df_texts["vader"]),
        "avg_sentiment_cc": safe_mean(df_texts[mask_cc]["vader"]),
        "avg_sentiment_non_cc": safe_mean(df_texts[mask_non_cc]["vader"]),
        
        # NER mention counts
        "total_ner_mentions": safe_sum(df_texts["ner_total"]),
        "total_ner_mentions_cc": safe_sum(df_texts[mask_cc]["ner_total"]),
        "total_ner_mentions_non_cc": safe_sum(df_texts[mask_non_cc]["ner_total"]),
        
        # Pronoun ratio
        "avg_pronoun_ratio": safe_mean(df_texts["pronoun_ratio"]),
        "avg_pronoun_ratio_cc": safe_mean(df_texts[mask_cc]["pronoun_ratio"]),
        "avg_pronoun_ratio_non_cc": safe_mean(df_texts[mask_non_cc]["pronoun_ratio"]),
        
        # Linguistic coordination
        "avg_linguistic_coordination": avg_coord,
        "avg_linguistic_coordination_cc": avg_coord_cc,
        "avg_linguistic_coordination_non_cc": avg_coord_non_cc
    }
    
    # Add Empath means for overall, CC, and non-CC
    for cat in TARGET_EMPATH_CATEGORIES:
        col = f"empath_{cat}"
        founder_profile[f"avg_empath_{cat}"] = safe_mean(df_texts[col])
        founder_profile[f"avg_empath_{cat}_cc"] = safe_mean(df_texts[mask_cc][col])
        founder_profile[f"avg_empath_{cat}_non_cc"] = safe_mean(df_texts[mask_non_cc][col])
        
    all_founders_profiles.append(founder_profile)

    # Save per-founder CSV
    founder_psy_path = os.path.join(founder_folder, f"{username}_psy_FEATURES.csv")
    df_single_founder = pd.DataFrame([founder_profile])
    df_single_founder.to_csv(founder_psy_path, index=False)

# Save global CSV
if len(all_founders_profiles) == len(founders):
    print("All founders processed.")
else:
    print(f"Processed {len(all_founders_profiles)} out of {len(founders)} founders.")

if all_founders_profiles:
    df_final = pd.DataFrame(all_founders_profiles)
    df_final.to_csv(path_founders_psy_out, index=False)
    print(f"Done. Aggregated profiles saved to: {path_founders_psy_out}")
else:
    print("No data extracted.")

