import requests
import pandas as pd
import time
import json
import os

matching_dict_path = "data/interim/matching_dict_optimized.json"
users_folder_path = "data/raw/FOUNDERS_DATA/"
FAILED_LOGS_DIR = "data/raw/FAILED_LOGS/"

os.makedirs(FAILED_LOGS_DIR, exist_ok=True)

base_url = "https://arctic-shift.photon-reddit.com/"

OBSERVATION_WINDOW_SECONDS = 3 * 30 * 24 * 60 * 60

foundersData_path = "data/interim/foundersData_FINAL.csv"
foundersData = pd.read_csv(foundersData_path, index_col=0)

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
# Collect founder usernames
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

print(f"Founder usernames: {founders}\n")

# Resume and retry helpers

def log_failure(username, filename):
    """Log failed username to a JSON file."""
    filepath = os.path.join(FAILED_LOGS_DIR, filename)
    failed_users = []
    
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                failed_users = json.load(f)
            except json.JSONDecodeError:
                failed_users = []
                
    if username not in failed_users:
        failed_users.append(username)
        with open(filepath, "w") as f:
            json.dump(failed_users, f, indent=4)

def fetch_with_retry(url, params, max_retries=5):
    """Run API request with retry on 50x errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=180)
            if response.status_code == 200:
                return response
            elif response.status_code in [500, 502, 503, 504]:
                print(f"Gateway Error {response.status_code}. Retry {attempt+1}/{max_retries}...")
                time.sleep(3)
            else:
                return response  # Non-retriable response
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}. Retry {attempt+1}/{max_retries}...")
            time.sleep(3)
            
    return None  # All retries failed


# Extraction blocks

# SUCCESFULLY EXTRACTED
# Extract posts and comments
def extract_posts_comments():
    
    posts_url = "/api/posts/search"
    comments_url = "/api/comments/search"

    # Iterate founders
    for username in founders:

        # Configure user output paths
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        os.makedirs(full_folder_path, exist_ok=True)

        path_posts_out = os.path.join(full_folder_path, username + "_posts.csv")
        path_comments_out = os.path.join(full_folder_path, username + "_comments.csv")

        # Resume check
        if os.path.exists(path_posts_out) and os.path.exists(path_comments_out):
            print(f"[{username}] Posts and comments already extracted; skipping.\n")
            continue

        # Configure observation window
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            log_failure(username, "failed_posts_comments.json")
            continue
            
        print(f"User row loaded for {username}.\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"Subreddit creation date: {subreddit_creation_date}\n")
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        user_success = True

        # Extract posts
        url = base_url + posts_url
        current_before_posts = before
        posts_data = []

        while True:
            params = {"author" : username, "before" : current_before_posts, "after" : after, "limit" : 100}
            response = fetch_with_retry(url, params=params)

            if not response or response.status_code != 200:
                code = response.status_code if response else "Timeout"
                print(f"HTTP error {code} on posts for {username}. Critical failure.\n")
                user_success = False
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

        if not user_success:
            log_failure(username, "failed_posts_comments.json")
            continue

        # Extract comments
        url = base_url + comments_url
        current_before_comments = before
        comments_data = []

        while True:
            params = {"author" : username, "before" : current_before_comments, "after" : after, "limit" : 100}
            response = fetch_with_retry(url, params=params)

            if not response or response.status_code != 200:
                code = response.status_code if response else "Timeout"
                print(f"HTTP error {code} on comments for {username}. Critical failure.\n")
                user_success = False
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

        if not user_success:
            log_failure(username, "failed_posts_comments.json")
            continue

        # Save user data
        df_posts = pd.DataFrame(posts_data)
        df_comments = pd.DataFrame(comments_data)
        
        df_posts.to_csv(path_posts_out, index=False)
        df_comments.to_csv(path_comments_out, index=False)

        time.sleep(1.0)

        if (posts_data and comments_data) or comments_data:
            print(f"Saved {len(posts_data)} posts by user {username}\n")
            print(f"Saved {len(comments_data)} comments by user {username}\n")


# NEED TO EXTRACT
# Extract interactions histogram
def extract_acquaintances_histograms():
    interactions_url = "/api/users/interactions/users"
    url = base_url + interactions_url

    for username in founders:

        # Configure user output path
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        os.makedirs(full_folder_path, exist_ok=True)
        
        path_interactions_histograms_out = os.path.join(full_folder_path, username + "_interactionsCount.csv")

        # Resume check
        if os.path.exists(path_interactions_histograms_out):
            print(f"[{username}] Interactions histogram already extracted; skipping.\n")
            continue

        # Configure observation window
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            log_failure(username, "failed_interactions_histograms.json")
            continue
            
        print(f"User row loaded for {username}.\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"Subreddit creation date: {subreddit_creation_date}\n")
        
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        params = {
            "author": username, 
            "before": before, 
            "after": after,   
            "limit" : 100000
        }
        
        response = fetch_with_retry(url=url, params=params) 

        if not response or response.status_code != 200:
            code = response.status_code if response else "Timeout"
            print(f"HTTP error {code} on interactions for {username}; skipping.\n")
            log_failure(username, "failed_interactions_histograms.json")
            continue

        responseData = response.json()
        interactions_data = responseData.get("data", [])
        
        if not interactions_data:
            print(f"No interactions found for {username} in the observation window.\n")
            # Save empty file to mark as processed
            pd.DataFrame().to_csv(path_interactions_histograms_out, index=False)
            continue

        df_interactions_histograms = pd.DataFrame(interactions_data)
        df_interactions_histograms.to_csv(path_interactions_histograms_out, index=False)

        print(f"Saved {len(interactions_data)} acquaintance interaction records for user {username}\n")
        
        time.sleep(3.0)  # Rate limit between users

# NEED TO EXTRACT
# Extract interactions list
def extract_interactions_list():
    interactions_list_url = "/api/users/interactions/users/list"
    url = base_url + interactions_list_url

    for username in founders:

        # Configure user output path
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        os.makedirs(full_folder_path, exist_ok=True)
        
        path_interactions_list_out = os.path.join(full_folder_path, username + "_interactionsList.csv")

        # Resume check
        if os.path.exists(path_interactions_list_out):
            print(f"[{username}] Interactions list already extracted; skipping.\n")
            continue

        # Configure observation window
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            log_failure(username, "failed_interactions_list.json")
            continue
            
        print(f"User row loaded for {username}.\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"Subreddit creation date: {subreddit_creation_date}\n")
        
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        params = {
            "author": username, 
            "before": before, 
            "after": after,   
            "limit" : 100000
        }
        
        response = fetch_with_retry(url=url, params=params) 

        if not response or response.status_code != 200:
            code = response.status_code if response else "Timeout"
            print(f"HTTP error {code} on interactions for {username}; skipping.\n")
            log_failure(username, "failed_interactions_list.json")
            continue

        responseData = response.json()
        interactions_list_data = responseData.get("data", [])
        
        if not interactions_list_data:
            print(f"No interactions found for {username} in the observation window.\n")
            pd.DataFrame().to_csv(path_interactions_list_out, index=False)
            continue

        df_interactions_list = pd.DataFrame(interactions_list_data)
        df_interactions_list.to_csv(path_interactions_list_out, index=False)

        print(f"Saved {len(interactions_list_data)} interactions for user {username}\n")
        
        time.sleep(3.0) 

# NEED TO EXTRACT
# Extract commented posts
def extract_posts_commented():

    postIDS_url = "/api/posts/ids"
    url = base_url + postIDS_url
    
    for username in founders:
        
        post_ids = set()

        # Configure user paths
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        
        path_posts_commented_out = os.path.join(full_folder_path, username + "_postsCommented.csv")

        # Resume check
        if os.path.exists(path_posts_commented_out):
            print(f"[{username}] Commented posts already extracted; skipping.\n")
            continue

        # Load comments dataset
        comments_path = full_folder_path + "/" + username + "_comments.csv"
        
        # Ensure dependency file exists
        if not os.path.exists(comments_path):
            print(f"[{username}] Cannot proceed: comments file is missing.\n")
            log_failure(username, "failed_posts_commented.json")
            continue

        comments = pd.read_csv(comments_path)
        if "link_id" not in comments.columns:
            pd.DataFrame().to_csv(path_posts_commented_out, index=False)
            continue

        for link_id in comments["link_id"].dropna():
            post_ids.add(str(link_id)[3:])
        
        post_ids_list = list(post_ids)
        posts_commented = []
        i = 0
        batch_size = 100  
        
        user_success = True

        while i < len(post_ids_list):
            batch = post_ids_list[i : i + batch_size]
            ids_parameter = ",".join(batch)

            params = {"ids" : ids_parameter}
            response = fetch_with_retry(url=url, params=params)
            
            if response and response.status_code == 200:
                try:
                    responseData = response.json()
                    batch_data = responseData.get("data", [])
                    posts_commented.extend(batch_data)
                except Exception:
                    print(f"JSON decode error on batch for {username}; server returned invalid payload.\n")
                    user_success = False
                    break
            else:
                code = response.status_code if response else "Timeout"
                print(f"HTTP error {code} on batch for {username}. Critical failure.\n")
                user_success = False
                break
            
            i += batch_size
            time.sleep(0.2) 
        
        if not user_success:
            log_failure(username, "failed_posts_commented.json")
            continue

        df_posts_commented = pd.DataFrame(posts_commented)
        df_posts_commented.to_csv(path_posts_commented_out, index=False)

        if posts_commented:
            print(f"Extracted {len(posts_commented)} posts commented by user {username}\n")

        time.sleep(1.0)

# NEED TO EXTRACT
# Extract received comments
def extract_comments_received():
    comments_url = "/api/comments/ids"
    url = base_url + comments_url

    for username in founders:
        
        comment_ids = set()

        # Configure user paths
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path

        path_comments_received_out = os.path.join(full_folder_path, username + "_commentsReceived.csv")

        # Resume check
        if os.path.exists(path_comments_received_out):
            print(f"[{username}] Received comments already extracted; skipping.\n")
            continue

        # Load interactions list dataset
        path_interactionsList = full_folder_path + "/" + username + "_interactionsList.csv"
        
        if not os.path.exists(path_interactionsList):
            print(f"[{username}] Cannot proceed: interactions list file is missing.\n")
            log_failure(username, "failed_comments_received.json")
            continue

        interactionsList = pd.read_csv(path_interactionsList)
        
        if "interaction_type" not in interactionsList.columns or "id" not in interactionsList.columns:
            pd.DataFrame().to_csv(path_comments_received_out, index=False)
            continue

        for id, interaction_type in zip(interactionsList["id"], interactionsList["interaction_type"]):
            if interaction_type == "commented under users post" or interaction_type == "comment under users comment":
                comment_ids.add(str(id)[3:])

        comment_ids_list = list(comment_ids)
        comments_received = []
        i = 0
        batch_size = 100 

        user_success = True

        while i < len(comment_ids_list):
            batch = comment_ids_list[i : i + batch_size]
            ids_parameter = ",".join(batch)

            params = {"ids" : ids_parameter}
            response = fetch_with_retry(url=url, params=params)
            
            if response and response.status_code == 200:
                try:
                    responseData = response.json()
                    batch_data = responseData.get("data", [])
                    comments_received.extend(batch_data)
                except Exception:
                    print(f"JSON decode error on batch for {username}.\n")
                    user_success = False
                    break
            else:
                code = response.status_code if response else "Timeout"
                print(f"HTTP error {code} on batch for {username}.\n")
                user_success = False
                break
            
            i += batch_size
            time.sleep(0.2) 
        
        if not user_success:
            log_failure(username, "failed_comments_received.json")
            continue

        df_comments_received = pd.DataFrame(comments_received)
        df_comments_received.to_csv(path_comments_received_out, index=False)

        if comments_received:
            print(f"Extracted {len(comments_received)} comments received for user {username}\n")

        time.sleep(1.0)

# NEED TO EXTRACT
# Extract acquaintances subreddits
def extract_acquaintances_subreddits():
    subreddits_url = "/api/users/interactions/subreddits"
    url = base_url + subreddits_url

    for username in founders:

        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        
        path_acquaintances_subreddits_out = os.path.join(full_folder_path, username + "_acquaintancesSubreddits.csv")

        # Resume check
        if os.path.exists(path_acquaintances_subreddits_out):
            print(f"[{username}] Acquaintance subreddits already extracted; skipping.\n")
            continue

        # Configure pre-observation window
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            log_failure(username, "failed_acquaintances_subreddits.json")
            continue
            
        print(f"User row loaded for {username}.\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"Subreddit creation date: {subreddit_creation_date}\n")
        before = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS
        
        acquaintances_subreddits = []

        # Load interactions count dataset
        path_interactionsCount = full_folder_path + "/" + username + "_interactionsCount.csv"
        
        if not os.path.exists(path_interactionsCount):
            print(f"[{username}] Cannot proceed: interactions count file is missing.\n")
            log_failure(username, "failed_acquaintances_subreddits.json")
            continue

        interactionsCount = pd.read_csv(path_interactionsCount)
        if "author" not in interactionsCount.columns:
            pd.DataFrame().to_csv(path_acquaintances_subreddits_out, index=False)
            continue

        acquaintances = interactionsCount["author"].dropna().tolist()

        if "[deleted]" in acquaintances:
            acquaintances.remove("[deleted]")
        if username in acquaintances:
            acquaintances.remove(username)

        user_has_errors = False

        for aq in acquaintances:

            params = {"author" : aq, "before" : before, "limit" : 100000}
            response = fetch_with_retry(url=url, params=params)
            
            aq_subreddits = [] 
            
            if response and response.status_code == 200:
                try:
                    responseData = response.json()
                    aq_subreddits = responseData.get("data", [])
                    print(f"Acquaintance {aq}: {len(aq_subreddits)} subreddits extracted.\n")
                except Exception:
                    print(f"JSON error for user {aq}.")
                    user_has_errors = True
            else:
                code = response.status_code if response else "Timeout"
                print(f"HTTP error {code} for user {aq}; skipping their subreddits.")
                user_has_errors = True

            aq_row = {"author" : aq, "subreddits" : aq_subreddits}
            acquaintances_subreddits.append(aq_row)

            time.sleep(0.2)
        
        # Save partial data even if some acquaintances fail
        df_acquaintances_subreddits = pd.DataFrame(acquaintances_subreddits)
        df_acquaintances_subreddits.to_csv(path_acquaintances_subreddits_out, index=False)

        if user_has_errors:
            log_failure(username, "failed_acquaintances_subreddits.json")

        if acquaintances_subreddits:
            print(f"Extracted subreddits for {len(acquaintances_subreddits)} acquaintances of user {username}\n")

        time.sleep(1.0)


if __name__ == "__main__":
    # Call needed functions here
    extract_acquaintances_histograms()