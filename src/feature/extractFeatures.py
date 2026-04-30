import requests 
import pandas as pd
import time

"""
for each founder, extract their matching features
dataset should have columns: 
founder name, account creation date, number of contributions, subreddit they're active on
"""


def extract_features():
    foundersFeatures = list()
    counter = 0
    base_url = "https://arctic-shift.photon-reddit.com/"
    users_url = "/api/users/search"
    subreddits_url = "/api/users/interactions/subreddits"

    df = pd.read_csv("data/raw/conspiracy_founders_clean.csv", index_col=0)
    
    for row in df.iloc[1500:].itertuples():
        username = row.author
        params = {
            "author" : username,
            "limit" : 1
        }

        url = base_url + users_url
        response = requests.get(url, params=params).json().get("data")
        responseData = response[0]
        founderMetaData = responseData.get("_meta")
        
        # Active Since
        if founderMetaData["earliest_comment_at"]:
            active_since = min(founderMetaData["earliest_comment_at"], founderMetaData["earliest_post_at"])
        else:
            active_since = founderMetaData["earliest_post_at"]

        # Number of Contributions
        if founderMetaData["num_comments"]:
            num_contributions = founderMetaData["num_posts"] + founderMetaData["num_comments"]
        else:
            num_contributions = founderMetaData["num_posts"]

        # Subreddits Active on
        params = {
            "author" : username,
            "limit" : 1000000
        }
        url = base_url + subreddits_url
        response = requests.get(url, params).json()
        founderSubredditsRaw = response.get("data")
        if founderSubredditsRaw:
            founderSubredditsClean = founderSubredditsRaw
        else:
            founderSubredditsClean = []

        # save data in new dataset
        founderData = {
            "username" : username,
            "active since" : active_since,
            "num contributions" : num_contributions,
            "subreddits" : founderSubredditsClean
        }

        print(f"{founderData}\n")
        foundersFeatures.append(founderData)
        counter += 1
        print(f"{counter}\n")
        time.sleep(0.1)
        
    # save dataset
    df_new = pd.DataFrame(foundersFeatures)
    df_new.to_csv("data/raw/founders_features.csv", mode='a', header=False)
    #df_new_section.to_csv("data/raw/founders_features_firstHalf.csv", mode='a', header=False)

if __name__ == "__main__":
    #extract_features()

    '''
    df = pd.read_csv("data/raw/founders_features.csv", index_col=0)
    first_part = df.iloc[:500]
    last_part = df.iloc[1000:]
    df_new = pd.concat([first_part, last_part])
    df_new.reset_index(drop=True, inplace=True)
    df_new.to_csv("data/raw/founders_features_firstHalf.csv")
    '''

    df = pd.read_csv("data/raw/founders_features.csv")
    df.reset_index(drop=True, inplace=True)
    df.to_csv("data/raw/founders_features.csv")