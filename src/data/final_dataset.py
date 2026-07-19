import pandas as pd
import json
import numpy as np
import os


users_folder_path = "data/raw/FOUNDERS_DATA/"
matching_dict_path = "data/interim/matching_dict_optimized.json"

foundersData_path = "data/interim/foundersData_FINAL.csv"
foundersFeatures_path = "data/interim/foundersFeatures_FINAL.csv"
foundersSubredditsObs_path = "data/raw/founders_subreddits_obs.csv"
foundersConspiracyScores_path = "data/interim/founders_conspiracy_scores.csv"

foundersData = pd.read_csv(foundersData_path)
foundersFeatures = pd.read_csv(foundersFeatures_path)
foundersSubredditsObs = pd.read_csv(foundersSubredditsObs_path)
foundersConspiracyScores = pd.read_csv(foundersConspiracyScores_path)

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates


def create_founder_row(username: str) -> dict:

    user_folder_path = username + "_data/"
    
    ratio_features_path = username + "_ratio_FEATURES.csv"
    conspiracy_features_path = username + "_conspiracySimilarity_FEATURES.csv"
    psy_features_path = username + "_psy_FEATURES.csv"
    
    foundersData_row = foundersData[foundersData["founders"] == username]
    foundersFeatures_row = foundersFeatures[foundersFeatures["username"] == username]
    foundersSubredditsObs_row = foundersSubredditsObs[foundersSubredditsObs["author"] == username]
    foundersConspiracyScores_row = foundersConspiracyScores[foundersConspiracyScores["author"] == username]

    subreddit_name = foundersData_row["subreddit_name"].iloc[0]
    subreddit_creation_date = foundersData_row["subreddit_creation_date"].iloc[0]
    
    active_since_unix = foundersFeatures_row["active since"].iloc[0]
    active_since_datetime = pd.to_datetime(active_since_unix, unit='s')
    
    total_contributions = foundersSubredditsObs_row["total_obs_contributions"].iloc[0]
    subreddits = foundersSubredditsObs_row["subreddits"].iloc[0]

    subreddits_conspiracy_score = foundersConspiracyScores_row["conspiracy_score"].iloc[0]

    # Calcolo manuale dei punteggi accumulati (Proxy temporale del Karma)
    raw_posts_path = os.path.join(users_folder_path, user_folder_path, f"{username}_posts.csv")
    raw_comments_path = os.path.join(users_folder_path, user_folder_path, f"{username}_comments.csv")
    
    accumulated_post_score = 0
    if os.path.exists(raw_posts_path):
        try:
            df_posts_raw = pd.read_csv(raw_posts_path)
            if "score" in df_posts_raw.columns:
                accumulated_post_score = int(df_posts_raw["score"].sum())
        except Exception:
            pass

    accumulated_comment_score = 0
    if os.path.exists(raw_comments_path):
        try:
            df_comments_raw = pd.read_csv(raw_comments_path)
            if "score" in df_comments_raw.columns:
                accumulated_comment_score = int(df_comments_raw["score"].sum())
        except Exception:
            pass

    accumulated_score = accumulated_post_score + accumulated_comment_score

    # Determine anchor status and matched companion
    if username in matching_dict:
        is_anchor = True
        user_matched = matching_dict[username]["candidate"]
    else:
        is_anchor = False
        # Find the anchor key that corresponds to this candidate
        user_matched = None
        for anchor_key, info in matching_dict.items():
            if info["candidate"] == username:
                user_matched = anchor_key
                break
    
    # create row
    row = {
        "username" : username,
        "is_anchor" : is_anchor,
        "user_matched" : user_matched,
        "active_since_unix" : active_since_unix,
        "active_since_datetime" : active_since_datetime,
        "subreddit_founded" : subreddit_name,
        "founding_date" : subreddit_creation_date,
        "total_contributions" : total_contributions,
        "subreddits" : subreddits,
        "subreddits_conspiracy_score" : subreddits_conspiracy_score,
        "accumulated_post_score" : accumulated_post_score,
        "accumulated_comment_score" : accumulated_comment_score,
        "accumulated_score" : accumulated_score
    }

    for feat_path in [conspiracy_features_path, ratio_features_path, psy_features_path]:
        path = users_folder_path + user_folder_path + feat_path
        df = pd.read_csv(path)
        features_dict = {str(k): v for k, v in df.iloc[0].to_dict().items()}
        features_dict.pop("author", None)
        features_dict.pop("total_contributions", None)
        row.update(features_dict)
    
    return row


def create_final_dataset():

    final_dataset = list()

    for founder in founders:
        founder_row = create_founder_row(founder)
        final_dataset.append(founder_row)

    df = pd.DataFrame(final_dataset)
    path_out = "data/processed/dataset.csv"
    
    # Assicura che la cartella finale esista prima di salvare
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    df.to_csv(path_out, index=False)
    print(f"Dataset unico generato con successo! Shape: {df.shape}")


if __name__ == "__main__":
    create_final_dataset()