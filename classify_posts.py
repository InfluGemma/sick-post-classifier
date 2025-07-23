from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax
import os, sys

path = os.path.dirname(os.path.abspath('./helper/states.py'))
if path not in sys.path:
    sys.path.append(path)


path = os.path.dirname(os.path.abspath('./classify/posts.py'))
if path not in sys.path:
    sys.path.append(path)

from posts import gather_posts

# Load trained model and tokenizer
model_name = "./post_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def classify_flu_posts(posts, batch_size=16):
    flu_related_count = 0
    total = len(posts)
    print(total)

    for i in range(0, total, batch_size):
        batch = posts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**encoded)
            probs = softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        flu_related_count += (preds == 1).sum().item()
    print(flu_related_count)
    flu_percent = flu_related_count / total * 100
    return flu_percent

posts = gather_posts("us", 9, "2022-04-02")

test = classify_flu_posts(posts)
print(test)
