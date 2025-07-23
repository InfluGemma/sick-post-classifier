import requests
from datetime import datetime, timedelta
import pandas as pd
import os, sys
import random


path = os.path.dirname(os.path.abspath('../helper/states.py'))
if path not in sys.path:
    sys.path.append(path)


from states import aus_states, us_regions
from subreddits import state_subreddits

# dates should be YYYY-MM-DD

def gather_posts(country, region, end):
    if country == "aus":
        states = aus_states
    else:
        states = us_regions[region]

    date_end = datetime.strptime(end, "%Y-%m-%d")
    date_start = date_end - timedelta(days=30)
    start = datetime.strftime(date_start, "%Y-%m-%d") + "T00%3A00"
    end = end + "T00%3A00"

    base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
    posts = []

    for state in states:
        sub = state_subreddits[state]
        url = base_url + "?subreddit=" + sub + "&after=" + start + "&before=" + end + "&limit=100"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json().get("data", [])

            for post in data:
                text = post.get("selftext", "")
                if text and text != "[removed]" and text != "[deleted]":
                    posts.append(text)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching from {sub}: {e}")

    return posts