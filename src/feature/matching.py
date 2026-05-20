'''Matching Method 1 (step by step per feature)'''

import pandas as pd

def subreddits_WMD(subreddits1: list, subreddits2: list) -> float:

    return 

def match_users(user1: dict, user2: dict) -> bool:

    # Account Creation Date
    if user1["active since"].year != user2["active since"].year:
        return False

    # Number of Contributions
    if user1["num contributions bin"] != user2["num contributions bin"]:
        return False
    
    # Subreddits Active on
    if subreddits_WMD(user1["subreddits"], user2["subreddits"]) > 0.3:
        return False

    return True
