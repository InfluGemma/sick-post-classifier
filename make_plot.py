import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("training-data/random_posts.csv", sep="|")
df = df.drop(['sub', 'year', 'symptoms'], axis=1)
df = df.dropna()
texts = df["text"].tolist()
y_true = df["label"].tolist()

# Load tokenizer and model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "/srv/scratch/z5397970/post_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Function to get predictions
def get_predictions(texts):
    preds = []
    for i in range(0, len(texts), 16):  # batch size of 16
        batch_texts = texts[i:i+16]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds.extend(batch_preds)
    return preds

# Run inference
y_pred = get_predictions(texts)

# Make confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Flu", "Flu"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("matrix2.png", dpi=300, bbox_inches="tight")
plt.close()
