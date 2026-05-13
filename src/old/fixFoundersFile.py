import pandas as pd

def fixFile():
    foundersFeatures = pd.read_csv("data/raw/founders_features_clean.csv", index_col=0)
    ogFile = pd.read_csv("data/raw/conspiracy_founders_cleanOUTDATED.csv", index_col=0)

    joinedFile = pd.merge(foundersFeatures, ogFile, how="inner", left_on="username", right_on="author")

    joinedFile.rename(columns={"active since" : "active_since",
                               "num contributions" : "num_contributions",
                               "title" : "post_title",
                               "name" : "subreddit_name",
                               "subreddits" : "subreddits_participated"}, inplace=True)
    
    joinedFile.drop(columns=["author", "active_since", "num_contributions", "subreddits_participated"], inplace=True)
    
    joinedFile.drop_duplicates("username", keep="first", inplace=True)
    joinedFile.reset_index(drop=True, inplace=True)
    
    print(joinedFile.info())
    print(joinedFile.tail(5))

    #joinedFile.to_csv("data/raw/conspiracy_founders_clean.csv")

if __name__ == "__main__":
    fixFile()