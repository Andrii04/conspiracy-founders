import requests
import pandas as pd
import time

'''
script che scarica la prima submission da arcticshift api 
per i subreddit nel vettore, introducendo un rate limit secondo specifiche, 
e li salva in data/raw (trovando founders)
Consideriamo i top 10% e bottom 10% subreddit del vettore come conspiracy/no, 
e corrispondentemente conspiracy/non-ct founder
'''


def findFounders():

    vectorSubreddits = pd.read_csv("data/external/conspiracy_svd_cossim_vector.csv", low_memory=False)
    vectorSubreddits.rename(columns={"Unnamed: 0" : "name"}, inplace=True)

    base_url = "https://arctic-shift.photon-reddit.com/"
    params = dict()
 
    result_data = list()
    for row in vectorSubreddits.itertuples():

        url = base_url + "api/subreddits/search"
        params = {
            "subreddit" : row.name,
            "fields" : "created_utc"
        }
        response = requests.get(url, params).json().get("data")
        if not response:
            continue
        subReddit_creationDate = response[0].get("created_utc")
        
        url = base_url + "api/posts/search"
        params = {
            "subreddit" : row.name,
            "sort" : "asc",
            "sort_type" : "created_utc",
            "limit" : 1,
            "fields" : "author, created_utc, id, title"
        }
        response = requests.get(url, params).json().get("data")
        if not response:
            continue
        post_data = {
            "author" : response[0].get("author"),
            "post_creation_date" : response[0].get("created_utc"),
            "id" : response[0].get("id"),
            "title" : response[0].get("title"),
            "original_index" : row.Index,
            "subreddit_creation_date" : subReddit_creationDate
        }
        print(f"Subreddit: {row.name}, Founder Data: {post_data}\n\n")
        result_data.append(post_data)

        time.sleep(0.5)

    subredditFirstPosts = pd.DataFrame(result_data)
    subredditFirstPosts.set_index("original_index", inplace=True)

    founders = vectorSubreddits.join(subredditFirstPosts, how="left")
    founders.to_csv("data/raw/conspiracy_founders.csv")


if __name__ == "__main__":
    findFounders()