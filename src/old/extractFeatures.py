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

    df = pd.read_csv("data/interim/founders.csv", index_col=0)
    
    for row in df.iloc[2000:].itertuples():
        username = row.founder
        params = {
            "author" : username,
            "limit" : 1
        }

        url = base_url + users_url
        response = requests.get(url, params=params).json().get("data")
        if response == None or len(response) == 0:
            continue
        responseData = response[0]
        founderMetaData = responseData.get("_meta")
        
        # Active Since
        c_time = founderMetaData.get("earliest_comment_at")
        p_time = founderMetaData.get("earliest_post_at")
        
        if c_time is not None and p_time is not None:
            active_since = min(c_time, p_time)
        elif c_time is not None:
            active_since = c_time
        elif p_time is not None:
            active_since = p_time
        else:
            active_since = None

        # Number of Contributions
        c_num = founderMetaData.get("num_comments") or 0
        p_num = founderMetaData.get("num_posts") or 0
        num_contributions = c_num + p_num

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
    df_new.to_csv("data/raw/real_founders_features.csv", mode='a', header=False)
    #df_new_section.to_csv("data/raw/founders_features_firstHalf.csv", mode='a', header=False)

if __name__ == "__main__":
    oldFeatures = pd.read_csv("data/interim/foundersFeatures_FINAL.csv", index_col=0)
    foundersData = pd.read_csv("data/interim/foundersData_FINAL.csv", index_col=0)

    print(foundersData.info())

    