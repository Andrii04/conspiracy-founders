import pandas as pd
# dropped rows: 16

def cleanFoundersFeatures():

    df = pd.read_csv("data/raw/founders_features.csv", index_col=0)

    # drop rows with empty subreddits list
    df_filtered = df[ df["subreddits"] != "[]" ]
    # drop duplicates
    df_filtered.drop_duplicates(subset="username", keep="first", inplace=True)

    df_filtered.reset_index(drop=True, inplace=True)
    df_filtered.drop(columns=["Unnamed: 0"], inplace=True)
    df_filtered.to_csv("data/raw/founders_features_clean.csv")

if __name__ == "__main__":
    cleanFoundersFeatures()