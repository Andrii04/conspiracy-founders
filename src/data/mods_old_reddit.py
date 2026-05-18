import csv
import json
import os
import random
import re
import time
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# replace with your actual reddit_session cookie value
reddit_session_cookie = os.getenv("reddit_session_cookie")
# replace with your actual browser user-agent string
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

headers = {
    "User-Agent": user_agent
}

cookies = {
    "reddit_session": reddit_session_cookie
}

session = requests.Session()
session.headers.update(headers)
session.cookies.update(cookies)


def fetch_moderators(subreddit_name):
    url = f"https://old.reddit.com/r/{subreddit_name}/about/moderators/.json"
    try:
        response = session.get(url, timeout=10)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"rate limited on {subreddit_name}. sleeping longer...")
            time.sleep(60)
            return fetch_moderators(subreddit_name)
        else:
            print(f"failed to fetch {subreddit_name}: status code {response.status_code}")
            return None
    except Exception as e:
        print(f"error fetching {subreddit_name}: {e}")
        return None

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
def main():
    input_file = "subredditDataWA.csv"

    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            subreddit_name = row["subreddit_name"]
            if not subreddit_name:
                continue
            fname = sanitize_filename(subreddit_name)
            if os.path.exists(f"../../data/interim/mods_json/{fname}_mods.json"):continue

            print(f"fetching moderators for r/{subreddit_name}...")
            data = fetch_moderators(subreddit_name)

            if data:
                with open(f"../../data/interim/mods_json/{fname}_mods.json", "w", encoding="utf-8") as out_f:
                    json.dump(data, out_f, indent=4)

            # remain undetected by introducing a random delay between 5 and 10 seconds
            delay = random.uniform(5, 10)
            time.sleep(delay)


if __name__ == "__main__":
    main()