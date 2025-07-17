import pandas
import os, sys
import random
from datetime import datetime, timedelta
import requests
import numpy as np

path = os.path.dirname(os.path.abspath('../helper/subreddits.py'))
if path not in sys.path:
    sys.path.append(path)

from subreddits import state_subreddits
state_subreddits = state_subreddits

posts = []
base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
symptoms = ['flu', 'influenza', 'fever', 'headache', 'sick', 'cough', 'symptom', 'virus']


# Grab 100 posts from 7 random subreddits
for i in range(0, 5):
    start_year = random.randint(2016, 2024)
    start_month = random.randint(1,12)
    start = datetime(start_year, start_month, 1)
    end = start + timedelta(days=60)

    start_str = start.strftime("%Y-%m-%d") + "T00%3A00"
    end_str = end.strftime("%Y-%m-%d") + "T00%3A00"

    sub = random.choice(list(state_subreddits.values()))
    # symptom = random.choice(list(symptoms))
    symptom = "flu"
    url = base_url + "?subreddit=" + sub + "&after=" + start_str + "&before=" + end_str + "&selftext=" + symptom + "&limit=20" 

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json().get("data", [])

        for post in data:
            text = post.get("selftext", "")
            if text and text != "[removed]" and text != "[deleted]":
                posts.append((text, sub, start_year))

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from {sub}")

# data = {"Text": posts[0], "Sub": posts[1]}
df = pandas.DataFrame(posts, columns=['text', 'sub', 'year'])
df['symptoms'] = df['text'].str.contains('|'.join(symptoms), na=False)
print(df)

# Add label for dataset
df["Label"] = 0

df.to_csv('random_posts.csv', mode='a', index=False, sep="|", header=False)