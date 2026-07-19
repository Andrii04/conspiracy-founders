import pandas as pd
import os
import json
import numpy as np
import ast
from config import OBSERVATION_WINDOW_SECONDS
from config import CC_THRESHOLD

# Config
users_folder_path = "data/raw/FOUNDERS_DATA/"
matching_dict_path = "data/interim/matching_dict_optimized.json"

OBS_SUBREDDITS_PATH = "data/raw/founders_subreddits_obs.csv"
PREOBS_SUBREDDITS_PATH = "data/interim/foundersFeatures_FINAL.csv"

# Helpers

def get_founders_list():
    """Return the founder list from JSON."""
    with open(matching_dict_path, "r") as f:
        matching_dict = json.load(f)
    anchors = list(matching_dict.keys())
    candidates = [candidate["candidate"] for candidate in matching_dict.values()]
    return anchors + candidates

def create_cc_map(df_acq):
    """Build {author: is_cc} from conspiracy score."""
    cc_map = {}
    if df_acq is not None and not df_acq.empty:
        for _, row in df_acq.iterrows():
            cc_map[str(row['author'])] = float(row['conspiracy_score']) > CC_THRESHOLD
    return cc_map

def count_moderated(df_posts, df_comments):
    """Count moderator-removed contributions."""
    mod_count = 0
    if df_comments is not None and 'body' in df_comments.columns:
        mod_count += df_comments['body'].isin(['[removed]']).sum()
    if df_posts is not None:
        if 'selftext' in df_posts.columns:
            mod_count += df_posts['selftext'].isin(['[removed]']).sum()
        if 'removed' in df_posts.columns:
            mod_count += (df_posts['removed'] == True).sum()
    return mod_count

def count_deleted(df_posts, df_comments):
    """Count user-deleted contributions."""
    del_count = 0
    if df_comments is not None and 'body' in df_comments.columns:
        del_count += df_comments['body'].isin(['[deleted]']).sum()
    if df_posts is not None and 'selftext' in df_posts.columns:
        del_count += df_posts['selftext'].isin(['[deleted]']).sum()
    return del_count

def count_negative(df_posts, df_comments):
    """Count contributions with negative score."""
    neg_count = 0
    if df_posts is not None and 'score' in df_posts.columns:
        neg_count += (pd.to_numeric(df_posts['score'], errors='coerce').fillna(0) < 0).sum()
    if df_comments is not None and 'score' in df_comments.columns:
        neg_count += (pd.to_numeric(df_comments['score'], errors='coerce').fillna(0) < 0).sum()
    return neg_count

def count_positive(df_posts, df_comments):
    """Count contributions with positive score."""
    pos_count = 0
    if df_posts is not None and 'score' in df_posts.columns:
        pos_count += (pd.to_numeric(df_posts['score'], errors='coerce').fillna(0) > 0).sum()
    if df_comments is not None and 'score' in df_comments.columns:
        pos_count += (pd.to_numeric(df_comments['score'], errors='coerce').fillna(0) > 0).sum()
    return pos_count

def safe_read_csv(filepath):
    """Read CSV if it exists, else return None."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None

def calculate_gini(subreddits_list):
    """Compute the Gini coefficient for subreddit histograms."""
    if not subreddits_list:
        return 0.0
    
    # Keep positive numeric counts only
    counts = [float(sub.get('count', 0)) for sub in subreddits_list if isinstance(sub, dict)]
    counts = [c for c in counts if c > 0]
    
    if len(counts) <= 1:
        # Gini is not informative with <= 1 value
        return 0.0
        
    # Standard Gini on sorted values
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    
    gini = ((np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts)))
    return gini

# Main feature extraction

def extract_other_features(username, obs_map, preobs_map):
    """Compute all requested ratios for one founder."""
    
    folder = os.path.join(users_folder_path, f"{username}_data")
    
    # Load user datasets
    df_acq = safe_read_csv(os.path.join(folder, f"{username}_acquaintancesScores.csv"))
    df_posts = safe_read_csv(os.path.join(folder, f"{username}_posts.csv"))
    df_comments = safe_read_csv(os.path.join(folder, f"{username}_comments.csv"))
    df_recv = safe_read_csv(os.path.join(folder, f"{username}_commentsReceived.csv"))
    df_posts_comm = safe_read_csv(os.path.join(folder, f"{username}_postsCommented.csv"))
    
    cc_map = create_cc_map(df_acq)
    
    # Posting vs commenting ratios
    len_posts = len(df_posts) if df_posts is not None else 0
    len_comments = len(df_comments) if df_comments is not None else 0
    total_contributions = len_posts + len_comments
    
    ratio_posting = len_posts / total_contributions if total_contributions > 0 else 0
    ratio_commenting = len_comments / total_contributions if total_contributions > 0 else 0
    
    # Global moderation/deletion/score ratios
    total_moderated = count_moderated(df_posts, df_comments)
    total_deleted = count_deleted(df_posts, df_comments)
    total_negative = count_negative(df_posts, df_comments)
    total_positive = count_positive(df_posts, df_comments)
    
    ratio_moderated = total_moderated / total_contributions if total_contributions > 0 else 0
    ratio_deleted = total_deleted / total_contributions if total_contributions > 0 else 0
    ratio_negative = total_negative / total_contributions if total_contributions > 0 else 0
    ratio_positive = total_positive / total_contributions if total_contributions > 0 else 0
    
    # Map target id to author
    target_author_map = {}
    if df_recv is not None:
        for _, row in df_recv.iterrows():
            target_author_map[str(row['name'])] = str(row['author'])
    if df_posts_comm is not None:
        for _, row in df_posts_comm.iterrows():
            target_author_map[str(row['name'])] = str(row['author'])

    # Outgoing interactions to CC vs non-CC
    to_cc_count = 0
    to_non_cc_count = 0
    
    if df_comments is not None and 'parent_id' in df_comments.columns:
        for _, row in df_comments.iterrows():
            parent_id = str(row['parent_id'])
            if parent_id in target_author_map:
                target_author = target_author_map[parent_id]
                # Check whether target author is CC
                if cc_map.get(target_author, False):
                    to_cc_count += 1
                else:
                    to_non_cc_count += 1
                    
    total_mapped_out = to_cc_count + to_non_cc_count
    ratio_user_to_cc = to_cc_count / total_mapped_out if total_mapped_out > 0 else 0
    ratio_user_to_non_cc = to_non_cc_count / total_mapped_out if total_mapped_out > 0 else 0

    # Incoming interactions from CC vs non-CC
    from_cc_count = 0
    from_non_cc_count = 0
    total_recv = len(df_recv) if df_recv is not None else 0
    
    if df_recv is not None and 'author' in df_recv.columns:
        for _, row in df_recv.iterrows():
            if cc_map.get(str(row['author']), False):
                from_cc_count += 1
            else:
                from_non_cc_count += 1
                
    ratio_user_from_cc = from_cc_count / total_recv if total_recv > 0 else 0
    ratio_user_from_non_cc = from_non_cc_count / total_recv if total_recv > 0 else 0
    
    # Combined interactions with CC vs non-CC
    total_cc_interactions = to_cc_count + from_cc_count
    total_non_cc_interactions = to_non_cc_count + from_non_cc_count
    total_all_interactions = total_cc_interactions + total_non_cc_interactions
    
    ratio_interactions_cc = total_cc_interactions / total_all_interactions if total_all_interactions > 0 else 0
    ratio_interactions_non_cc = total_non_cc_interactions / total_all_interactions if total_all_interactions > 0 else 0
    
    # Receiving ratios over total activity
    total_activity = total_contributions + total_recv
    ratio_recv_global = total_recv / total_activity if total_activity > 0 else 0
    ratio_recv_cc = from_cc_count / total_activity if total_activity > 0 else 0
    ratio_recv_non_cc = from_non_cc_count / total_activity if total_activity > 0 else 0

    # Thread coverage by CC vs non-CC
    all_threads = set()
    cc_threads = set()
    non_cc_threads = set()
    
    # Collect founder thread ids
    if df_posts is not None and 'name' in df_posts.columns:
        all_threads.update(df_posts['name'].dropna().astype(str))
    if df_comments is not None and 'link_id' in df_comments.columns:
        all_threads.update(df_comments['link_id'].dropna().astype(str))
        
    # Mark threads with CC or non-CC presence
    if df_posts_comm is not None:
        for _, row in df_posts_comm.iterrows():
            thread_id = str(row['name'])
            if cc_map.get(str(row['author']), False):
                cc_threads.add(thread_id)
            else:
                non_cc_threads.add(thread_id)
                
    if df_recv is not None:
        for _, row in df_recv.iterrows():
            thread_id = str(row['link_id'])
            if cc_map.get(str(row['author']), False):
                cc_threads.add(thread_id)
            else:
                non_cc_threads.add(thread_id)

    # Intersections for ratio computation
    founder_cc_threads = all_threads.intersection(cc_threads)
    founder_non_cc_threads = all_threads.intersection(non_cc_threads)
    
    ratio_threads_with_cc = len(founder_cc_threads) / len(all_threads) if len(all_threads) > 0 else 0
    ratio_threads_with_non_cc = len(founder_non_cc_threads) / len(all_threads) if len(all_threads) > 0 else 0

    # Gini and growing exclusivity
    obs_subs = obs_map.get(username, [])
    preobs_subs = preobs_map.get(username, [])
    
    gini_obs = calculate_gini(obs_subs)
    gini_preobs = calculate_gini(preobs_subs)
    growing_exclusivity = gini_obs - gini_preobs

    # Build output row
    return {
        "author": username,
        "total_contributions": total_contributions,
        
        # Outbound ratios
        "ratio_posting": ratio_posting,
        "ratio_commenting": ratio_commenting,
        
        # Combined interaction ratios
        "ratio_interactions_cc": ratio_interactions_cc,
        "ratio_interactions_non_cc": ratio_interactions_non_cc,
        
        # Outgoing interaction ratios
        "ratio_user_to_cc_interactions": ratio_user_to_cc,
        "ratio_user_to_non_cc_interactions": ratio_user_to_non_cc,
        "ratio_commenting_under_cc": ratio_user_to_cc,  # Backward compatibility
        "ratio_commenting_under_non_cc": ratio_user_to_non_cc,
        
        # Incoming interaction ratios
        "ratio_user_from_cc_interactions": ratio_user_from_cc,
        "ratio_user_from_non_cc_interactions": ratio_user_from_non_cc,
        
        # Receiving ratios over total activity
        "ratio_receiving_comments": ratio_recv_global,
        "ratio_receiving_comments_cc": ratio_recv_cc,
        "ratio_receiving_comments_non_cc": ratio_recv_non_cc,
        
        # Thread ratios
        "ratio_threads_with_cc": ratio_threads_with_cc,
        "ratio_threads_with_non_cc": ratio_threads_with_non_cc,
        
        # Global moderation/deletion/score ratios
        "ratio_moderated_contributions": ratio_moderated,
        "ratio_deleted_contributions": ratio_deleted,
        "ratio_negative_contributions": ratio_negative,
        "ratio_positive_contributions": ratio_positive,
        
        # Subreddit exclusivity (Gini)
        "subreddits_gini_obs": gini_obs,
        "subreddits_gini_preobs": gini_preobs,
        "growing_exclusivity_contributions": growing_exclusivity
    }

# Script entry point

if __name__ == "__main__":
    founders = get_founders_list()
    
    print("Loading global histogram datasets (obs and pre-obs)...")
    
    # Load obs dataset
    obs_map = {}
    if os.path.exists(OBS_SUBREDDITS_PATH):
        df_obs = pd.read_csv(OBS_SUBREDDITS_PATH)
        for _, row in df_obs.iterrows():
            try:
                # Parse serialized subreddit list
                subs = ast.literal_eval(row['subreddits']) if isinstance(row['subreddits'], str) else row['subreddits']
                obs_map[str(row['author'])] = subs
            except Exception:
                obs_map[str(row['author'])] = []
    else:
        print(f"Warning: file not found '{OBS_SUBREDDITS_PATH}'")

    # Load pre-obs dataset
    preobs_map = {}
    if os.path.exists(PREOBS_SUBREDDITS_PATH):
        df_preobs = pd.read_csv(PREOBS_SUBREDDITS_PATH)
        for _, row in df_preobs.iterrows():
            try:
                subs = ast.literal_eval(row['subreddits']) if isinstance(row['subreddits'], str) else row['subreddits']
                preobs_map[str(row['username'])] = subs
            except Exception:
                preobs_map[str(row['username'])] = []
    else:
        print(f"Warning: file not found '{PREOBS_SUBREDDITS_PATH}'")
        
    print(f"Datasets loaded (obs: {len(obs_map)} users, pre-obs: {len(preobs_map)} users). Starting feature extraction...")
    
    for username in founders:
        print(f"Extracting ratio features for: {username}")
        
        try:
            features = extract_other_features(username, obs_map, preobs_map)
            
            # Save per-user CSV
            out_path = os.path.join(users_folder_path, f"{username}_data", f"{username}_ratio_FEATURES.csv")
            df_out = pd.DataFrame([features])
            df_out.to_csv(out_path, index=False)
            
            print(f"Saved: {username}_ratio_FEATURES.csv")
            
        except Exception as e:
            print(f"Error processing {username}: {e}")
            
    print("Feature extraction completed for all founders.")