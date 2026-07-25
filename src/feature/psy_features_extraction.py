import pandas as pd
import spacy
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
import os
import json

CC_THRESHOLD = 0.15

# Initialize NLP tools
print("Loading NLP models...")
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()
empath = Empath()

# Minimized raw Empath categories needed for calculations (including custom equations components)
TARGET_EMPATH_CATEGORIES = [
    "fear", "violence", "government", "politics", "power", "deception", "business", "leader",
    "suffering", "pain", "torment", "neglect", "weakness", "shame", "magic", "legend",
    "hipster", "science", "school", "college", "negative_emotion", "nervousness",
    "sadness", "anger", "horror", "cheerfulness", "optimism", "friends", "party",
    "leisure", "communication", "play", "vacation", "sleep", "anonymity", "philosophy", "art",
    "reading", "independence", "domestic_work", "politeness", "sympathy", "trust", "affection",
    "family", "healing", "aggression", "hate", "swearing_terms", "fight", "rage", "ridicule",
    "dominant_personality", "work", "order", "law", "achievement", "office", "childish",
    "disappointment", "pride", "irritability", "envy", "exasperation", "disgust", "contentment",
    "positive_emotion"
]

# Standalone categories that must keep individual columns
STANDALONE_EMPATH_CATEGORIES = [
    "fear", "violence", "power", "deception", "anger", "sadness", "nervousness",
    "dominant_personality", "pride", "optimism", "shame", "irritability", "trust",
    "politeness", "sympathy", "cheerfulness", "negative_emotion", "aggression",
    "hate", "swearing_terms", "envy", "exasperation", "contentment", "positive_emotion"
]

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

# Helpers per la pulizia dei dati e lettura sicura
def safe_read_csv(filepath):
    """Legge un file CSV in modo sicuro gestendo file inesistenti, vuoti (0 byte) o corrotti."""
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
    if pd.isna(val): return ""
    return str(val).strip()

def is_valid_text(text):
    s = str(text).strip()
    return bool(s) and s not in ["[deleted]", "[removed]"]

def get_vader_compound(text):
    if not is_valid_text(text): return 0.0
    return vader.polarity_scores(str(text))["compound"]

def get_empath_scores(text):
    if not is_valid_text(text): 
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    scores = empath.analyze(str(text), normalize=True)
    if not scores:
        return {cat: 0.0 for cat in TARGET_EMPATH_CATEGORIES}
    return {cat: scores.get(cat, 0.0) for cat in TARGET_EMPATH_CATEGORIES}

def get_spacy_features(text):
    if not is_valid_text(text): 
        return 0, 0, 0.0
    doc = nlp(str(text))
    
    ner_count = len([ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]])
    
    total_tokens = len(doc)
    if total_tokens == 0: return ner_count, 0, 0.0
    pronoun_count = len([token for token in doc if token.pos_ == "PRON"])
    pronoun_ratio = pronoun_count / total_tokens
    
    return ner_count, pronoun_count, pronoun_ratio

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
    
    acq_path = os.path.join(founder_folder, username + "_acquaintancesScores.csv")
    recv_path = os.path.join(founder_folder, username + "_commentsReceived.csv")
    posts_comm_path = os.path.join(founder_folder, username + "_postsCommented.csv")
    coord_path = os.path.join(founder_folder, username + "_linguisticCoordination.csv")
    
    # Build CC map for acquaintances
    cc_map = {}
    df_acq = safe_read_csv(acq_path)
    if df_acq is not None and 'author' in df_acq.columns and 'conspiracy_score' in df_acq.columns:
        for _, row in df_acq.iterrows():
            cc_map[clean_id(row['author'])] = float(row['conspiracy_score']) > CC_THRESHOLD
                
    # Build target-author map for replies
    target_author_map = {}
    df_recv = safe_read_csv(recv_path)
    if df_recv is not None and 'name' in df_recv.columns and 'author' in df_recv.columns:
        for _, row in df_recv.iterrows():
            target_author_map[clean_id(row['name'])] = clean_id(row['author'])

    df_p_comm = safe_read_csv(posts_comm_path)
    if df_p_comm is not None and 'name' in df_p_comm.columns and 'author' in df_p_comm.columns:
        for _, row in df_p_comm.iterrows():
            target_author_map[clean_id(row['name'])] = clean_id(row['author'])

    texts_data = []
    
    # Load and merge comment texts with CC tag
    df_c = safe_read_csv(comments_path)
    if df_c is not None and 'body' in df_c.columns and 'parent_id' in df_c.columns:
        for _, row in df_c.iterrows():
            body = str(row['body']) if pd.notna(row['body']) else ""
            if not is_valid_text(body): continue
            
            parent_id = clean_id(row['parent_id'])
            target_author = target_author_map.get(parent_id)
            is_cc = cc_map.get(target_author, False) if target_author else False
            
            texts_data.append({"text": body, "is_cc": is_cc})
                
    # Load and merge post texts (ora gestito in modo sicuro contro file vuoti)
    df_p = safe_read_csv(posts_path)
    if df_p is not None and "title" in df_p.columns and "selftext" in df_p.columns:
        for _, row in df_p.iterrows():
            title = str(row['title']) if pd.notna(row['title']) else ""
            body = str(row['selftext']) if pd.notna(row['selftext']) else ""
            text = f"{title} {body}".strip()
            if not is_valid_text(text): continue
            
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
    
    # Compute linguistic coordination (Global, Positive, Negative)
    avg_coord = 0.0
    avg_coord_cc = 0.0
    avg_coord_non_cc = 0.0
    
    avg_pos_coord = 0.0
    avg_pos_coord_cc = 0.0
    avg_pos_coord_non_cc = 0.0
    
    avg_neg_coord = 0.0
    avg_neg_coord_cc = 0.0
    avg_neg_coord_non_cc = 0.0
    
    df_coord = safe_read_csv(coord_path)
    if df_coord is not None and 'is_cc' in df_coord.columns:
        mask_coord_cc = df_coord['is_cc'] == True
        mask_coord_non_cc = df_coord['is_cc'] == False
        
        if 'coordination_score' in df_coord.columns:
            avg_coord = safe_mean(df_coord['coordination_score'])
            avg_coord_cc = safe_mean(df_coord[mask_coord_cc]['coordination_score'])
            avg_coord_non_cc = safe_mean(df_coord[mask_coord_non_cc]['coordination_score'])
            
        if 'positive_coordination_score' in df_coord.columns:
            avg_pos_coord = safe_mean(df_coord['positive_coordination_score'])
            avg_pos_coord_cc = safe_mean(df_coord[mask_coord_cc]['positive_coordination_score'])
            avg_pos_coord_non_cc = safe_mean(df_coord[mask_coord_non_cc]['positive_coordination_score'])
            
        if 'negative_coordination_score' in df_coord.columns:
            avg_neg_coord = safe_mean(df_coord['negative_coordination_score'])
            avg_neg_coord_cc = safe_mean(df_coord[mask_coord_cc]['negative_coordination_score'])
            avg_neg_coord_non_cc = safe_mean(df_coord[mask_coord_non_cc]['negative_coordination_score'])
    
    # Aggregate base founder profile
    founder_profile = {
        "author": username,
        "total_contributions": len(df_texts),
        
        # Sentiment
        "avg_sentiment": safe_mean(df_texts["vader"]),
        "avg_sentiment_cc": safe_mean(df_texts[mask_cc]["vader"]),
        "avg_sentiment_non_cc": safe_mean(df_texts[mask_non_cc]["vader"]),
        
        # NER mentions
        "total_ner_mentions": safe_sum(df_texts["ner_total"]),
        "total_ner_mentions_cc": safe_sum(df_texts[mask_cc]["ner_total"]),
        "total_ner_mentions_non_cc": safe_sum(df_texts[mask_non_cc]["ner_total"]),
        
        # Pronouns
        "avg_pronoun_ratio": safe_mean(df_texts["pronoun_ratio"]),
        "avg_pronoun_ratio_cc": safe_mean(df_texts[mask_cc]["pronoun_ratio"]),
        "avg_pronoun_ratio_non_cc": safe_mean(df_texts[mask_non_cc]["pronoun_ratio"]),
        
        # Coordination
        "avg_linguistic_coordination": avg_coord,
        "avg_linguistic_coordination_cc": avg_coord_cc,
        "avg_linguistic_coordination_non_cc": avg_coord_non_cc,
        
        "avg_positive_linguistic_coordination": avg_pos_coord,
        "avg_positive_linguistic_coordination_cc": avg_pos_coord_cc,
        "avg_positive_linguistic_coordination_non_cc": avg_pos_coord_non_cc,
        
        "avg_negative_linguistic_coordination": avg_neg_coord,
        "avg_negative_linguistic_coordination_cc": avg_neg_coord_cc,
        "avg_negative_linguistic_coordination_non_cc": avg_neg_coord_non_cc
    }
    
    # 1. Calculate temporary raw averages for all categories
    raw_means = {}
    raw_means_cc = {}
    raw_means_non_cc = {}
    for cat in TARGET_EMPATH_CATEGORIES:
        col = f"empath_{cat}"
        raw_means[cat] = safe_mean(df_texts[col])
        raw_means_cc[cat] = safe_mean(df_texts[mask_cc][col])
        raw_means_non_cc[cat] = safe_mean(df_texts[mask_non_cc][col])

    # 2. Compute custom category metrics
    custom_metrics = {}
    suffixes = [("", raw_means), ("_cc", raw_means_cc), ("_non_cc", raw_means_non_cc)]
    
    for suff, source in suffixes:
        v = lambda c: source.get(c, 0.0)
        
        custom_metrics[f"avg_distrust{suff}"] = v("government") + v("politics") + v("power") + v("deception") + v("business") + v("leader")
        custom_metrics[f"avg_victim_complex{suff}"] = v("suffering") + v("pain") + v("torment") + v("neglect") + v("weakness") + v("shame")
        custom_metrics[f"avg_epistemic_non_conformity{suff}"] = (v("magic") + v("legend") + v("hipster")) - (v("science") + v("school") + v("college"))
        
        custom_metrics[f"avg_neuroticism{suff}"] = v("negative_emotion") + v("nervousness") + v("sadness") + v("anger") + v("suffering") + v("horror") + v("fear") + v("irritability") + v("exasperation") + v("disgust") - v("cheerfulness") - v("optimism")
        custom_metrics[f"avg_extraversion{suff}"] = v("friends") + v("party") + v("cheerfulness") + v("leisure") + v("communication") + v("play") + v("vacation") - v("sleep") - v("anonymity")
        custom_metrics[f"avg_openness{suff}"] = v("philosophy") + v("art") + v("science") + v("magic") + v("hipster") + v("legend") + v("reading") + v("independence") - v("domestic_work")
        custom_metrics[f"avg_agreeableness{suff}"] = v("politeness") + v("sympathy") + v("trust") + v("affection") + v("family") + v("healing") - v("aggression") - v("hate") - v("swearing_terms") - v("violence") - v("fight") - v("rage") - v("ridicule") - v("dominant_personality")
        custom_metrics[f"avg_conscientousness{suff}"] = v("work") + v("order") + v("law") + v("achievement") + v("business") + v("school") + v("office") - v("play") - v("leisure") - v("childish")
        
        # Combined unified Self-Esteem metric
        custom_metrics[f"avg_self_esteem{suff}"] = (v("pride") + v("achievement") + v("optimism")) - (v("shame") + v("weakness") + v("disappointment") + v("envy"))

    founder_profile.update(custom_metrics)
    
    for cat in STANDALONE_EMPATH_CATEGORIES:
        founder_profile[f"avg_{cat}"] = raw_means[cat]
        founder_profile[f"avg_{cat}_cc"] = raw_means_cc[cat]
        founder_profile[f"avg_{cat}_non_cc"] = raw_means_non_cc[cat]
        
    all_founders_profiles.append(founder_profile)

    founder_psy_path = os.path.join(founder_folder, f"{username}_psy_FEATURES.csv")
    df_single_founder = pd.DataFrame([founder_profile])
    df_single_founder.to_csv(founder_psy_path, index=False)

if len(all_founders_profiles) == len(founders):
    print("All founders processed.")
else:
    print(f"Processed {len(all_founders_profiles)} out of {len(founders)} founders.")

if all_founders_profiles:
    df_final = pd.DataFrame(all_founders_profiles)
    df_final.to_csv(path_founders_psy_out, index=False)
    print(f"Aggregated profiles saved to: {path_founders_psy_out}")
else:
    print("No data extracted.")
