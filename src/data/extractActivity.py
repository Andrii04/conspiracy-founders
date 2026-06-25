import requests
import pandas as pd
import time
import json
import os



matching_dict_path = "data/interim/matching_dict_optimized.json"
users_folder_path = "data/raw/FOUNDERS_DATA/"

base_url = "https://arctic-shift.photon-reddit.com/"

OBSERVATION_WINDOW_SECONDS = 3 * 30 * 24 * 60 * 60

foundersData_path = "data/interim/foundersData_FINAL.csv"
foundersData = pd.read_csv(foundersData_path, index_col=0)

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)
    
# get all founders usernames
anchors = [anchor for anchor in matching_dict.keys()]
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates

print(f"founder usernames: {founders}\n")



# SUCCESSFULLY EXTRACTED
def extract_posts_comments():
    
    posts_url = "/api/posts/search"
    comments_url = "/api/comments/search"

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



# NEED TO EXTRACT
def extract_acquaintances_histograms():
    interactions_url = "/api/users/interactions/users"
    url = base_url + interactions_url

    for username in founders:

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        
        # observation window threshold configuration
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            continue
            
        print(f"Extracted user row: {userRow}\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"subreddit creation date: {subreddit_creation_date}\n")
        
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        params = {
            "author": username, 
            "before": before, 
            "after": after,   
            "limit" : 100000
        }
        
        response = requests.get(url=url, params=params) 

        if response.status_code != 200:
            print(f"Errore HTTP {response.status_code}: {response.reason} su interazioni di {username}. Salto.\n")
            continue

        responseData = response.json()
        interactions_data = responseData.get("data", [])
        
        if not interactions_data:
            print(f"Nessuna interazione trovata per {username} nel trimestre.\n")
            continue

        df_interactions_histograms = pd.DataFrame(interactions_data)
        path_interactions_histograms_out = os.path.join(full_folder_path, username + "_interactionsCount.csv")

        df_interactions_histograms.to_csv(path_interactions_histograms_out, index=False)

        print(f"Saved {len(interactions_data)} acquaintance interaction records for user {username}\n")
        
        time.sleep(3.0) # Cortesia per il server tra un utente e l'altro



# NEED TO EXTRACT
def extract_interactions_list():
    interactions_list_url = "/api/users/interactions/users/list"
    url = base_url + interactions_list_url

    for username in founders:

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path
        
        # observation window threshold configuration
        userRow = foundersData[foundersData["founder"] == username]
        if userRow.empty:
            continue
            
        print(f"Extracted user row: {userRow}\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"subreddit creation date: {subreddit_creation_date}\n")
        
        before = subreddit_creation_date
        after = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS

        params = {
            "author": username, 
            "before": before, 
            "after": after,   
            "limit" : 100000
        }
        
        response = requests.get(url=url, params=params) 

        if response.status_code != 200:
            print(f"Errore HTTP {response.status_code}: {response.reason} su interazioni di {username}. Salto.\n")
            continue

        responseData = response.json()
        interactions_list_data = responseData.get("data", [])
        
        if not interactions_list_data:
            print(f"Nessuna interazione trovata per {username} nel trimestre.\n")
            continue

        df_interactions_list = pd.DataFrame(interactions_list_data)
        path_interactions_list_out = os.path.join(full_folder_path, username + "_interactionsList.csv")

        df_interactions_list.to_csv(path_interactions_list_out, index=False)

        print(f"Saved {len(interactions_list_data)} interactions for user {username}\n")
        
        time.sleep(3.0) # Cortesia per il server tra un utente e l'altro



# NEED TO EXTRACT
def extract_posts_commented():

    postIDS_url = "/api/posts/ids"
    url = base_url + postIDS_url
    
    for username in founders:
        
        post_ids = set()

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path

        # load *founder*_comments.csv dataset
        comments_path = full_folder_path + "/" + username + "_comments.csv"
        comments = pd.read_csv(comments_path)

        for link_id in comments["link_id"]:
            post_ids.add(link_id[3:])
        

        post_ids_list = list(post_ids)
        posts_commented = []
        i = 0
        batch_size = 100  # Dimensione massima per batch per evitare URL troppo lunghi

        while i < len(post_ids_list):
            batch = post_ids_list[i : i + batch_size]
            ids_parameter = ",".join(batch)

            params = {"ids" : ids_parameter}
            response = requests.get(url=url, params=params)
            

            if response.status_code == 200:
                try:
                    responseData = response.json()
                    batch_data = responseData.get("data", [])
                    posts_commented.extend(batch_data)
                except Exception:
                    print(f"Errore di decodifica JSON su un batch di {username}. Il server ha risposto con HTML.\n")
            else:
                print(f"Errore HTTP {response.status_code} su un batch di {username}. Salto questo frammento.\n")
            
            i += batch_size
            time.sleep(0.2)  # Piccola pausa di cortesia tra i batch dello stesso utente
        

        df_posts_commented = pd.DataFrame(posts_commented)
        path_posts_commented_out = os.path.join(full_folder_path, username + "_postsCommented.csv")

        df_posts_commented.to_csv(path_posts_commented_out, index=False)

        if posts_commented:
            print(f"Extracted {len(posts_commented)} posts commented by user {username}\n")

        time.sleep(1.0)



# NEED TO EXTRACT
def extract_comments_received():
    comments_url = "/api/comments/ids"
    url = base_url + comments_url

    for username in founders:
        
        comment_ids = set()

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path

        # load *founder*_interactionsList.csv dataset
        path_interactionsList = full_folder_path + "/" + username + "_interactionsList.csv"
        interactionsList = pd.read_csv(path_interactionsList)

        for id, interaction_type in zip(interactionsList["id"], interactionsList["interaction_type"]):
            
            if interaction_type == "commented under users post" or interaction_type == "comment under users comment":
                comment_ids.add(id[3:])


        comment_ids_list = list(comment_ids)
        comments_received = []
        i = 0
        batch_size = 100  # Dimensione massima per batch per evitare URL troppo lunghi

        while i < len(comment_ids_list):
            batch = comment_ids_list[i : i + batch_size]
            ids_parameter = ",".join(batch)

            params = {"ids" : ids_parameter}
            response = requests.get(url=url, params=params)
            
            # Controllo di sicurezza sulla response per evitare crash con errori 504/HTML
            if response.status_code == 200:
                try:
                    responseData = response.json()
                    batch_data = responseData.get("data", [])
                    comments_received.extend(batch_data)
                except Exception:
                    print(f"Errore di decodifica JSON su un batch di {username}. Il server ha risposto con HTML.\n")
            else:
                print(f"Errore HTTP {response.status_code} su un batch di {username}. Salto questo frammento.\n")
            
            i += batch_size
            time.sleep(0.2)  # Piccola pausa di cortesia tra i batch dello stesso utente
        

        df_comments_received = pd.DataFrame(comments_received)
        path_comments_received_out = full_folder_path + "/" + username + "_commentsReceived.csv"

        df_comments_received.to_csv(path_comments_received_out, index=False)

        if comments_received:
            print(f"Extracted {len(comments_received)} comments received for user {username}\n")

        time.sleep(1.0)



# NEED TO EXTRACT
def extract_acquaintances_subreddits():
    subreddits_url = "/api/users/interactions/subreddits"
    url = base_url + subreddits_url

    for username in founders:

        # pre-observation window threshold configuration
        userRow = foundersData[foundersData["founder"] == username]
        print(f"Extracted user row: {userRow}\n")
        subreddit_creation_date = userRow["subreddit_creation_date"].values[0]
        print(f"subreddit creation date: {subreddit_creation_date}\n")
        before = subreddit_creation_date - OBSERVATION_WINDOW_SECONDS
        
        acquaintances_subreddits = []

        # user folder path configuration
        founder_folder_path = username + "_data"
        full_folder_path = users_folder_path + founder_folder_path

        # load *founder*_interactionsCount.csv dataset
        path_interactionsCount = full_folder_path + "/" + username + "_interactionsCount.csv"
        interactionsCount = pd.read_csv(path_interactionsCount)

        acquaintances = interactionsCount["author"].tolist()

        if "[deleted]" in acquaintances:
            acquaintances.remove("[deleted]")
        if username in acquaintances:
            acquaintances.remove(username)

        for aq in acquaintances:

            params = {"author" : aq, "before" : before, "limit" : 100000}
            response = requests.get(url=url, params=params)
            

            aq_subreddits = [] # Valore di default vuoto se la chiamata fallisce
            
            if response.status_code == 200:
                try:
                    responseData = response.json()
                    aq_subreddits = responseData.get("data", [])
                    print(f"extracting acquaintance {aq} subreddits, size: {len(aq_subreddits)}\n")
                except Exception:
                    print(f"Errore JSON per l'utente {aq}. Il server ha risposto con HTML.")
            else:
                print(f"Errore HTTP {response.status_code} per l'utente {aq}. Salto i suoi subreddits.")
   

            aq_row = {"author" : aq, "subreddits" : aq_subreddits}
            acquaintances_subreddits.append(aq_row)

            time.sleep(0.2)
        
        df_acquaintances_subreddits = pd.DataFrame(acquaintances_subreddits)
        path_acquaintances_subreddits_out = full_folder_path + "/" + username + "_acquaintancesSubreddits.csv"

        df_acquaintances_subreddits.to_csv(path_acquaintances_subreddits_out, index=False)

        if acquaintances_subreddits:
            print(f"Extracted subreddits for {len(acquaintances_subreddits)} acquaintances of user {username}\n")

        time.sleep(1.0)



if __name__ == "__main__":
    True

