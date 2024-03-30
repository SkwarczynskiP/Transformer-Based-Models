# from datasets import load_dataset
# from datasets import Dataset
# from transformers import AutoTokenizer
# from transformers import AutoModelForSequenceClassification
# from transformers import TrainingArguments, Trainer
# import numpy as np
# import evaluate

# def tokenizeBert(text):
#     return tokenizerBert(text["sms"], padding="max_length", truncation=True, max_length=512)
#
#
# def tokenizeT5(text):
#     return tokenizerT5(text["sms"], padding="max_length", truncation=True, max_length=512)
#
#
# def computeMetrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# dataset = load_dataset("sms_spam")
#
# tokenizerBert = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
# tokenizedDatasetBert = dataset.map(tokenizeBert, batched=True)
# trainDatasetBert = tokenizedDatasetBert['train'].select(range(4456))  # Left out the .shuffle
# testDatasetBert = tokenizedDatasetBert['train'].select(range(1114))  # Left out the .shuffle
# modelBert = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
# BertTraining_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
#
# trainerBert = Trainer(
#     model=modelBert,
#     args=BertTraining_args,
#     train_dataset=trainDatasetBert,
#     eval_dataset=testDatasetBert,
#     compute_metrics=computeMetrics,
# )
#
# tokenizerT5 = AutoTokenizer.from_pretrained("t5-small")
# tokenizedDatasetT5 = dataset.map(tokenizeT5, batched=True)
# trainDatasetT5 = tokenizedDatasetT5['train'].select(range(4456))  # Left out the .shuffle
# testDatasetT5 = tokenizedDatasetT5['train'].select(range(1114))  # Left out the .shuffle
# modelT5 = AutoModelForSequenceClassification.from_pretrained("t5-small", num_labels=2)
# T5Training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
#
# trainerT5 = Trainer(
#     model=modelT5,
#     args=T5Training_args,
#     train_dataset=trainDatasetT5,
#     eval_dataset=testDatasetT5,
#     compute_metrics=computeMetrics,
# )
#
# metric = evaluate.load("accuracy")
#
# trainerBert.train()
# trainerT5.train()

# ABOVE IS USEFUL CODE



# BELOW IS TESTING CODE FROM THE WEBSITE
# import torch
#
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
#
# # dataset = load_dataset("yelp_review_full")
# subset_train_dataset = load_dataset("yelp_review_full", split='train[:1000]')
# subset_eval_dataset = load_dataset("yelp_review_full", split='test[:1000]')
#
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
#
#
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
# # tokenized_dataset = dataset.map(tokenize_function, batched=True)
#
# tokenized_train_dataset = subset_train_dataset.map(tokenize_function, batched=True)
# tokenized_eval_dataset = subset_eval_dataset.map(tokenize_function, batched=True)
#
# # small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
# # small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))
#
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# metric = evaluate.load("accuracy")
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# # trainer = Trainer(
# #     model=model,
# #     args=training_args,
# #     train_dataset=small_train_dataset,
# #     eval_dataset=small_eval_dataset,
# #     compute_metrics=compute_metrics,
# # )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_train_dataset,
#     eval_dataset=tokenized_eval_dataset,
#     compute_metrics=compute_metrics,
# )
#
#
# trainer.train()

# ABOVE IS TESTING CODE FROM THE WEBSITE




# BELOW IS TESTING CODE FROM THE COPILOT
# import torch
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())
#
# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print("There are %d GPU(s) available." % torch.cuda.device_count())
#     print("We will use the GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device('cpu')
#
# from datasets import load_dataset
# from datasets import Dataset
# from transformers import BertTokenizer
# from transformers import AutoModelForSequenceClassification
# from transformers import TrainingArguments, Trainer
# import numpy as np
# import evaluate
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#
#
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
# dataset = load_dataset("yelp_review_full")
#
# tokenized_dataset = dataset.map(tokenize_function, batched=True)
#
# small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))
#
# model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
#
# model = model.to(device)
#
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
#
# metric = evaluate.load("accuracy")
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
# trainer.train()



#ANOTHER ATTEMPT


import evaluate
from datasets import load_dataset, metric
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import nltk
import collections
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", trust_remote_code=True)
tokenizerOne = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizerTwo = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")


def tokenize_function_one(examples):
    return tokenizerOne(examples["comment_text"], padding="max_length", truncation=True)


def tokenize_function_two(examples):
    return tokenizerTwo(examples["comment_text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess_text(text):
    # Convert text to lower
    text = text.lower()

    # Remove Punctuation
    tokenizer = RegexpTokenizer(r'\w+\'?\w+|\w+')
    words = tokenizer.tokenize(text)
    words_without_punctuation = [''.join(c for c in word if c not in string.punctuation or c in ["'", "’"]) for
                                 word in words]
    text = ' '.join(words_without_punctuation)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmaWords = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
    text = ' '.join(lemmaWords)

    # Remove stop words
    stop_words_lower = set(word.lower() for word in stopwords.words('english'))
    stop_words_upper = set(word.title() for word in stopwords.words('english'))
    stop_words = stop_words_lower.union(stop_words_upper)
    tokens = word_tokenize(text)
    tokensNoSWs = [tok for tok in tokens if tok not in stop_words]
    processedText = ' '.join(tokensNoSWs)

    # Remove numbers
    processedText = re.sub(r'\d+', '', processedText)

    # Remove unimportant and common words, and singular letters from the text
    processedText = processedText.replace("reuters", "")
    processedText = processedText.replace("ap", "")
    processedText = processedText.replace("'s ", "")
    processedText = processedText.replace("washingtonpost", "")
    processedText = processedText.replace(" say ", "")
    processedText = processedText.replace(" new ", "")
    processedText = processedText.replace(" lt ", "")
    processedText = processedText.replace(" gt ", "")
    processedText = re.sub(r'\b\w\b', '', processedText)

    return processedText


# START OF "MAIN" FUNCTION

# 2.2 Fine-tuning pre-trained models

# Model One: google-bert/bert-base-cased
print("Model One: google-bert/bert-base-cased...")
tokenized_datasets_one = dataset.map(tokenize_function_one, batched=True)
small_train_dataset_one = tokenized_datasets_one["balanced_train"].shuffle(seed=42).select(range(1000))
small_eval_dataset_one = tokenized_datasets_one["test"].shuffle(seed=42).select(range(1000))
model_one = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")
trainer_one = Trainer(
    model=model_one,
    args=training_args,
    train_dataset=small_train_dataset_one,
    eval_dataset=small_eval_dataset_one,
    compute_metrics=compute_metrics,
)
trainer_one.train()

# Next four lines of code were written by GitHub Copilot
# This code is meant to predict the labels for the test split and then print the accuracy
predictions = trainer_one.predict(small_eval_dataset_one)
class_predictions = np.argmax(predictions.predictions, axis=-1)
eval_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(eval_metrics)
print("\n")

# Model Two: distilbert/distilbert-base-multilingual-cased
print("Model Two: distilbert/distilbert-base-multilingual-cased...")
tokenized_datasets_two = dataset.map(tokenize_function_two, batched=True)
small_train_dataset_two = tokenized_datasets_two["balanced_train"].shuffle(seed=42).select(range(1000))
small_eval_dataset_two = tokenized_datasets_two["test"].shuffle(seed=42).select(range(1000))
model_two = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-multilingual-cased",
                                                               num_labels=2)
trainer_two = Trainer(
    model=model_two,
    args=training_args,
    train_dataset=small_train_dataset_two,
    eval_dataset=small_eval_dataset_two,
    compute_metrics=compute_metrics,
)
trainer_two.train()

# Next four lines of code were written by GitHub Copilot
# This code is meant to predict the labels for the test split and then print the accuracy
predictions = trainer_two.predict(small_eval_dataset_two)
class_predictions = np.argmax(predictions.predictions, axis=-1)
eval_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(eval_metrics)
print("\n")


