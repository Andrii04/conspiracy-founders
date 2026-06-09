import pandas as pd
import requests
import os
import json
import time
import re

def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use as a filename/directory name.
    Replaces characters that are invalid on Windows/Unix filesystems.
    """
    # Replace invalid characters with underscore
    # Windows invalid chars: < > : " / \ | ? *
    # Also replace control characters (0-31) and DEL (127)
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f\x7f]'
    sanitized = re.sub(invalid_chars, '_', name)

    # Remove leading/trailing spaces and dots (problematic on Windows)
    sanitized = sanitized.strip('. ')

    # Ensure the name is not empty
    if not sanitized:
        sanitized = 'unnamed'

    return sanitized

def get_founder(row):
    subreddit_name = row.subreddit_name
    subreddit_start=row.subreddit_creation_date
    fpath = f'/home/andri/conspiracy-founders/data/interim/mods_json/{sanitize_filename(subreddit_name)}_mods.json'
    if os.path.exists(fpath):
        with open(fpath, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        for mod in json_data.get("data", dict()).get("children", []):
            mod_start = mod.get("date", time.time())
            if mod_start<=subreddit_start+60*60*1: #1h since creation
                return mod.get("name")
    return


if __name__ == "__main__":
    df=pd.read_csv("/home/andri/conspiracy-founders/data/interim/subredditDataWA.csv", index_col=0)
    df["founder"]=df.apply(get_founder, axis=1)
    df.dropna(subset=["founder"], inplace=True)
    print(df.info())
    df.to_csv("/home/andri/conspiracy-founders/data/interim/founders.csv")