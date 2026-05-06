import pandas as pd
# dropped rows: 16

def cleanFoundersFeatures():

    df = pd.read_csv("data/raw/founders_features.csv")

    # drop rows with empty subreddits list
    df_filtered = df[ df["subreddits"] != "[]" ]
    print(f"dropped rows with empty subreddit list: {len(df) - len(df_filtered)}\n")

    df_filtered.to_csv("data/raw/founders_features.csv")

if __name__ == "__main__":
    cleanFoundersFeatures()