import pandas
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

df = pandas.read_csv("training-data/random_posts.csv", sep="|")
df = df.drop(['sub', 'year', 'symptoms'], axis=1)

# Undersample 
df_minority = df[df['label'] == 1]
df_majority = df[df['label'] == 0].sample(n=len(df_minority)*2, random_state=42)  # 2:1 ratio
df_balanced = pandas.concat([df_minority, df_majority])

# Split data into test and train sets
df_train, df_test = train_test_split(df_balanced, test_size=0.2)

# Convert to dataset object
train = Dataset.from_pandas(df_train)
test = Dataset.from_pandas(df_test)

#Tokenize data
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_train = train.map(tokenize, batched=True)
tokenized_test = test.map(tokenize, batched=True)

# Set up evalutation
# accuracy = evaluate.load("accuracy")

def metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = numpy.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Labels
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=metrics,
)

trainer.train()

trainer.save_model("./post_classifier")