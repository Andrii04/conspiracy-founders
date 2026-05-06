import pandas as pd

# script che pulisce il file cv conspiracy_founders_raw.csv togliendo righe con valori nan o founders bannati
# cambiato anche "id" a "postid" per evitare confusione

# DELETED ROWS: 1175
# NAN VALUES DELETED: 170
# [deleted] AUTHOR ROWS REMOVED: 1005

def cleanFoundersRaw():

    foundersRaw = pd.read_csv("data/raw/conspiracy_founders_raw.csv", index_col=0)

    num_rows_initially = len(foundersRaw)
    nan_values = foundersRaw["author"].isnull().sum()
    deleted_values = (foundersRaw["author"] == "[deleted]").sum()

    foundersRaw.dropna(subset="author", inplace=True)
    foundersClean = foundersRaw[foundersRaw["author"] != "[deleted]"]

    foundersClean.rename(columns={"id" : "postId"}, inplace=True)
    foundersClean.reset_index(drop=True, inplace=True)

    foundersClean.to_csv("data/raw/conspiracy_founders_clean.csv")

    print(f"deleted rows: {num_rows_initially - len(foundersClean)}\n")
    print(f"nan values deleted: {nan_values}\n")
    print(f"[deleted] authors rows removed: {deleted_values}\n")


if __name__ == "__main__":
    cleanFoundersRaw()