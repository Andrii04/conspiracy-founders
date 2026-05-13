import requests
import pandas as pd
import time

def extractFields():

    conspiracyFounders_csv = pd.read_csv("data/raw/conspiracy_founders_clean.csv", index_col=0)
    base_url = "https://arctic-shift.photon-reddit.com/"
    result_data = list()
    
    count = 0
    for row in conspiracyFounders_csv.itertuples():

        #extract subreddit data
        url = base_url + "api/subreddits/search"
        subreddit_name = row.subreddit_name
        params = {
            "subreddit" : subreddit_name,
            "fields" : "earliest_post, earliest_comment, created_utc"
        }

        response = requests.get(url, params).json()
        print("subreddit dates: ")
        print(response)
        print('\n')
        response = response.get("data")
        response = response[0]
        fields = response.get("_meta")

        subreddit_earliest_comment = fields["earliest_comment"]
        subreddit_earliest_post = fields["earliest_post"]
        subreddit_created_utc = response.get("created_utc")


        #extract our founder's first comment in subreddit
        url = base_url + "api/comments/search"
        founder_name = row.username
        params = {
            "subreddit" : subreddit_name,
            "author" : founder_name,
            "sort" : "asc",
            "limit" : 1,
            "fields" : "created_utc"
        }
        
        response = requests.get(url, params).json()
        print("our founder first comment: ")
        print(response)
        print('\n')
        response = response.get("data")
        if response:
            founder_earliest_comment = response[0].get("created_utc")
        else:
            founder_earliest_comment = 0

        #extract author and date of first comment in subreddit
        params = {
            "subreddit" : subreddit_name,
            "sort" : "asc",
            "limit" : 1,
            "fields" : "author, created_utc"
        }
        response = requests.get(url, params).json()
        print("actual earliest comment and author: ")
        print(response)
        response = response.get("data") 

        if response:
            response = response[0]
            actual_earliest_comment = response.get("created_utc")
            first_comment_author = response.get("author")
        else:
            actual_earliest_comment = ""
            first_comment_author = ""


        data = {
            "subreddit_name" : subreddit_name,
            "subreddit_earliest_comment" : subreddit_earliest_comment,
            "subreddit_earliest_post" : subreddit_earliest_post,
            "subreddit_creation_date" : subreddit_created_utc,
            "ourFounder_earliest_comment" : founder_earliest_comment,
            "subreddit_earliest_post_author" : founder_name,
            "subreddit_earliest_comment_author" : first_comment_author
        }

        result_data.append(data)
        print("full data: ")
        print(data)
        print("\n")
        print(count)
        print("\n")
        count += 1
        time.sleep(0.1)

    

    df = pd.DataFrame(result_data)
    df.to_csv("all_dates.csv")

if __name__ == "__main__":
    extractFields()