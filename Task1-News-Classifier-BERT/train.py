from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Load Dataset
dataset = load_dataset("ag_news")
print("Dataset loaded ✅")

# 2. Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")
print("Tokenization done ✅")

# 3. Load Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=4
)

# 4. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# 5. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=100,
    optim="adamw_torch",
    dataloader_drop_last=True,
)

# 6. Train
small_train = tokenized_dataset["train"].shuffle(seed=42).select(range(10000))
small_test  = tokenized_dataset["test"].shuffle(seed=42).select(range(2000))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_test,
    compute_metrics=compute_metrics,
)

trainer.train()

# 7. Evaluate
results = trainer.evaluate()
print(f"Accuracy : {results['eval_accuracy'] * 100:.2f}%")
print(f"F1 Score : {results['eval_f1']:.4f}")

# 8. Save
cpu_model = model.cpu()
cpu_model.save_pretrained("./bert-news-classifier")
tokenizer.save_pretrained("./bert-news-classifier")
print("Model saved ✅")
