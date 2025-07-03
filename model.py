import pandas
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy
import evaluate

df_RHMD = pandas.read_csv("training-data/RHMD_Cleaned.csv")
df_random = pandas.read_csv("training-data/random_posts.csv")

# Combine two data sources to one set
frames = [df_RHMD, df_random]
df = pandas.concat(frames)

# Randomly shuffle data
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Split data into test and train sets
df_train, df_test = train_test_split(df_shuffled, test_size=0.2)

# Convert to dataset object
train = Dataset.from_pandas(df_train)
test = Dataset.from_pandas(df_test)

#Tokenize data
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = train.map(tokenize, batched=True)
tokenized_test = test.map(tokenize, batched=True)

# Set up evalutation
accuracy = evaluate.load("accuracy")

def metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = numpy.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Labels
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id)

training_args = TrainingArguments(
    output_dir="post_classifier",
    eval_strategy="epoch",
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