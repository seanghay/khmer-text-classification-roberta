from huggingface_hub import login
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
  AutoTokenizer, 
  AutoModelForSequenceClassification,
  Trainer,
  TrainingArguments,
  DataCollatorWithPadding
)

# authenticate
hf_token = None

if hf_token: 
  login(hf_token, add_to_git_credential=True)

# args
do_train=True
output_dir = "./outputs/khmer-text-classification-roberta"
dataset_name = "seanghay/khmer-categorized-news-60k"
model_name = "xlm-roberta-base"
text_column_name = "content"
seed = 42

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# download dataset
raw_datasets = load_dataset(dataset_name, split="train", use_auth_token=True)
raw_datasets = raw_datasets.shuffle(seed=seed)
raw_datasets = raw_datasets.train_test_split(test_size=0.1)
labels = raw_datasets['train'].features['label'].names

id2label = dict([(id, label) for id, label in enumerate(labels)])
label2id = dict([(label, id) for id, label in enumerate(labels)])

# download model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    id2label=id2label, 
    label2id=label2id,
    num_labels=len(labels)
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def preprocessor(examples):
    return tokenizer(examples[text_column_name], truncation=True)

vectorized_datasets = raw_datasets.map(preprocessor, batched=True, remove_columns=["id", "content", "title"])

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=vectorized_datasets["train"],
  eval_dataset=vectorized_datasets["test"],
  tokenizer=tokenizer,
  data_collator=data_collator,
  compute_metrics=compute_metrics,
)

if do_train:
  trainer.train()
  model.save_pretrained(trainer.args.output_dir)
  tokenizer.save_pretrained(trainer.args.output_dir)

if training_args.push_to_hub:  
  trainer.push_to_hub()