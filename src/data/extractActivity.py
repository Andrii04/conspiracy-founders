import requests
import pandas as pd
import time
import json
import os


matching_dict_path = "data/interim/matching_dict_optimized.json"
users_folder_path = "data/raw/FOUNDERS_DATA/"

base_url = "https://arctic-shift.photon-reddit.com/"

posts_url = "/api/posts/search"
comments_url = "/api/comments/search"

OBSERVATION_WINDOW_SECONDS = 3 * 30 * 24 * 60 * 60

foundersData_path = "data/interim/foundersData_FINAL.csv"
foundersData = pd.read_csv(foundersData_path, index_col=0)


def extractActivity():

    with open(matching_dict_path, "r") as f:
        matching_dict = json.load(f)
    
    # get all founders usernames
    anchors = [anchor for anchor in matching_dict.keys()]
    candidates = [candidate["candidate"] for candidate in matching_dict.values()]
    founders = anchors + candidates
    print(f"founder usernames: {founders}\n")
    
    # main loop
    for username in founders:

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        os.makedirs(full_folder_path, exist_ok=True)

        # observation window threshold configuration
        userRow = foundersData[foundersData["founder"] == username]
        print(f"Extracted user row: {userRow}\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"subreddit creation date: {subreddit_creation_date}\n")
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        # POSTS
        #posts_fields = "author, author_fullname, author_flair_text, created_utc, distinguished, id, retrieved_on, subreddit, subreddit_id, score, crosspost_parent, link_flair_text, num_comments, over_18, post_hint, spoiler, title, selftext"
        url = base_url + posts_url
        current_before_posts = before
        posts_data = []

        while True:
            params = {"author" : username, "before" : current_before_posts, "after" : after, "limit" : 100}
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Errore HTTP {response.status_code} sui commenti di {username}. Salto questo batch.\n")
                break
            
            responseData = response.json()
            batch_posts = responseData.get("data", [])

            if not batch_posts:
                break

            posts_data.extend(batch_posts)

            if len(batch_posts)<100:
                break

            current_before_posts = batch_posts[-1]["created_utc"] - 1
            time.sleep(0.2)

        df_posts = pd.DataFrame(posts_data)
        path_posts_out = os.path.join(full_folder_path, username + "_posts.csv")

        # COMMENTS
        #comments_fields = "author, author_fullname, author_flair_text, created_utc, distinguished, id, retrieved_on, subreddit, subreddit_id, score, body, link_id, parent_id"
        url = base_url + comments_url
        current_before_comments = before
        comments_data = []

        while True:
            params = {"author" : username, "before" : current_before_comments, "after" : after, "limit" : 100}
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Errore HTTP {response.status_code} sui commenti di {username}. Salto questo batch.\n")
                break
            
            responseData = response.json()
            batch_comments = responseData.get("data", [])

            if not batch_comments:
                break
            
            comments_data.extend(batch_comments)

            if len(batch_comments)<100:
                break

            current_before_comments = batch_comments[-1]["created_utc"] - 1
            time.sleep(0.2)

        df_comments = pd.DataFrame(comments_data)
        path_comments_out = os.path.join(full_folder_path, username + "_comments.csv")

        # save user data
        df_posts.to_csv(path_posts_out, index=False)
        df_comments.to_csv(path_comments_out, index=False)

        time.sleep(1.0)

        if (posts_data and comments_data) or comments_data:
            print(f"Saved {len(posts_data)} posts by user {username}\n")
            print(f"Saved {len(comments_data)} comments by user {username}\n")


if __name__ == "__main__":
    extractActivity()