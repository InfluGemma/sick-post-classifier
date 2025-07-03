import pandas
import os, sys
import random
import requests
from subreddits import state_subreddits
state_subreddits = state_subreddits

posts = []
base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"

# Grab 100 posts from 7 random subreddits
for i in range(0, 6):
    sub = random.choice(list(state_subreddits.values()))
    url = base_url + "?subreddit=" + sub + "&limit=100"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])

        for post in data:
            text = post.get("selftext", "")
            if text and text != "[removed]":
                posts.append(text)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from {sub}")

data = {"Text": posts}
df = pandas.DataFrame(data)

# Add label for dataset
df["Label"] = 0

df.to_csv('random_posts.csv', index=False)