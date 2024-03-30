from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate


def tokenizeBert(text):
    return tokenizerBert(text["sms"], padding="max_length", truncation=True, max_length=256)


def tokenizeRoberta(text):
    return tokenizerRoberta(text["sms"], padding="max_length", truncation=True, max_length=256)


def computeMetrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 2.1 Dataset
dataset = load_dataset("sms_spam", trust_remote_code=True)


# 2.2 Fine-tuning Pre-trained Models
# Model 1: prajjwal1/bert-tiny
print("\n\nModel 1: prajjwal1/bert-tiny")
tokenizerBert = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizedDatasetBert = dataset.map(tokenizeBert, batched=True)
trainDatasetBert = tokenizedDatasetBert['train'].shuffle(seed=42).select(range(4456))  # 80% Split of the dataset
testDatasetBert = tokenizedDatasetBert['train'].shuffle(seed=42).select(range(1114))  # 20% Split of the dataset
modelBert = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)
BertTraining_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

# Creation of Bert Trainer Object
trainerBert = Trainer(
    model=modelBert,
    args=BertTraining_args,
    train_dataset=trainDatasetBert,
    eval_dataset=testDatasetBert,
    compute_metrics=computeMetrics,
)

# Fine-tuning the Model
trainerBert.train()

predictionsBert = trainerBert.predict(testDatasetBert)
classPredictionsBert = np.argmax(predictionsBert.predictions, axis=-1)
evalMetricsBert = computeMetrics((predictionsBert.predictions, predictionsBert.label_ids))
print(evalMetricsBert)

# Model 2: distilroberta-base
print("\n\nModel 2: distilroberta-base")
tokenizerRoberta = AutoTokenizer.from_pretrained("distilroberta-base")
tokenizedDatasetRoberta = dataset.map(tokenizeRoberta, batched=True)
trainDatasetRoberta = tokenizedDatasetRoberta['train'].shuffle(seed=42).select(range(4456))  # 80% Split of the dataset
testDatasetRoberta = tokenizedDatasetRoberta['train'].shuffle(seed=42).select(range(1114))  # 20% Split of the dataset
modelRoberta = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
RobertaTraining_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

# Creation of T5 Trainer Object
trainerRoberta = Trainer(
    model=modelRoberta,
    args=RobertaTraining_args,
    train_dataset=trainDatasetRoberta,
    eval_dataset=testDatasetRoberta,
    compute_metrics=computeMetrics,
)

# Fine-tuning the Model
trainerRoberta.train()

predictionsRoberta = trainerRoberta.predict(testDatasetRoberta)
classPredictionsRoberta = np.argmax(predictionsRoberta.predictions, axis=-1)
evalMetricsRoberta = computeMetrics((predictionsRoberta.predictions, predictionsRoberta.label_ids))
print(evalMetricsRoberta)


# 2.3 Zero-Shot Classification
# This is done on the Google Collab Notebook


# 2.4 Baselines
