import pandas as pd
import os
import json
import numpy as np

# Config
users_folder_path = "data/raw/FOUNDERS_DATA/"
matching_dict_path = "data/interim/matching_dict_optimized.json"

with open(matching_dict_path, "r") as f:
    matching_dict = json.load(f)

anchors = list(matching_dict.keys())
candidates = [candidate["candidate"] for candidate in matching_dict.values()]
founders = anchors + candidates