from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import numpy as np
import evaluate
import re
import nltk


def tokenize_bert(text):
    return tokenizerBert(text["sms"], padding="max_length", truncation=True, max_length=256)


def tokenize_roberta(text):
    return tokenizerRoberta(text["sms"], padding="max_length", truncation=True, max_length=256)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess(text):  # Function used in previous assignments
    text = text.lower()  # Lowercases each token
    text = text.replace("\\n", "")  # Gets rid of the "\n" characters in the dataset

    # Removes all punctuation (referenced https://www.geeksforgeeks.org/python-remove-punctuation-from-string/)
    text = re.sub(r'[^\w\s]', '', text)

    # Removes all stopwords
    stopWords = set(stopwords.words('english'))
    tempList = text.split()
    filteredWords = [tempList for tempList in tempList if tempList.lower() not in stopWords]
    text = (' '.join(filteredWords))

    # Lemmatization of each token (referenced
    # https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/ for different examples)
    lemmatizer = WordNetLemmatizer()
    tempList = text.split()
    lemma = [lemmatizer.lemmatize(tempList) for tempList in tempList]
    finalText = (' '.join(lemma))

    return finalText


# 2.1 Dataset
dataset = load_dataset("sms_spam", trust_remote_code=True)


# 2.2 Fine-tuning Pre-trained Models
# Referenced https://huggingface.co/docs/transformers/training for implementation of fine-tuning the models
# Also referenced Microsoft Copilot & numerous other HuggingFace documentation for help troubleshooting the code not
#   automatically executing on my laptop's GPU:
#   - https://discuss.huggingface.co/t/not-using-gpu-although-it-is-specified/14746
#   - https://discuss.huggingface.co/t/use-gpu-on-pc-to-load-the-models/604

# Model 1: prajjwal1/bert-tiny
print("\n\nModel 1: prajjwal1/bert-tiny")
tokenizerBert = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
tokenizedDatasetBert = dataset.map(tokenize_bert, batched=True)
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
    compute_metrics=compute_metrics,
)

# Fine-tuning the Model
trainerBert.train()

predictionsBert = trainerBert.predict(testDatasetBert)
classPredictionsBert = np.argmax(predictionsBert.predictions, axis=-1)
evalMetricsBert = compute_metrics((predictionsBert.predictions, predictionsBert.label_ids))
print(evalMetricsBert)

# Model 2: distilroberta-base
print("\n\nModel 2: distilroberta-base")
tokenizerRoberta = AutoTokenizer.from_pretrained("distilroberta-base")
tokenizedDatasetRoberta = dataset.map(tokenize_roberta, batched=True)
trainDatasetRoberta = tokenizedDatasetRoberta['train'].shuffle(seed=42).select(range(4456))  # 80% Split of the dataset
testDatasetRoberta = tokenizedDatasetRoberta['train'].shuffle(seed=42).select(range(1114))  # 20% Split of the dataset
modelRoberta = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=2)
RobertaTraining_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

# Creation of the distilroberta Trainer Object
trainerRoberta = Trainer(
    model=modelRoberta,
    args=RobertaTraining_args,
    train_dataset=trainDatasetRoberta,
    eval_dataset=testDatasetRoberta,
    compute_metrics=compute_metrics,
)

# Fine-tuning the Model
trainerRoberta.train()

predictionsRoberta = trainerRoberta.predict(testDatasetRoberta)
classPredictionsRoberta = np.argmax(predictionsRoberta.predictions, axis=-1)
evalMetricsRoberta = compute_metrics((predictionsRoberta.predictions, predictionsRoberta.label_ids))
print(evalMetricsRoberta)


# 2.3 Zero-Shot Classification
# Code for Zero-Shot Classification is found in the Zero-Shot Classification.ipynb file
# Referenced Microsoft Copilot for the code implementation, as many models were having trouble with the particular
#   dataset.


# 2.4 Baselines
# Preprocessing the Training Portion of the Dataset
textTrain = dataset['train']['sms'][0:4456]
labelsTrain = dataset['train']['label'][0:4456]
processedSentencesTrain = []

for text in textTrain:
    processedTextTrain = preprocess(text)
    processedSentencesTrain.append(processedTextTrain)

# Creating the Bag of Words Model For the Train Portion of the Dataset (Code taken from previous assignments)
vocab = set()
bowTrain = []
for text in processedSentencesTrain:
    wordCounts = {}
    tokens = nltk.word_tokenize(text)
    vocab.update(tokens)

    for word in tokens:
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1

    bowTrain.append(wordCounts)

# Preprocessing the Test Portion of the Dataset
textTest = dataset['train']['sms'][4456:5570]
labelsTest = dataset['train']['label'][4456:5570]
processedSentencesTest = []

for text in textTest:
    processedTextTest = preprocess(text)
    processedSentencesTest.append(processedTextTest)

# Creating the Bag of Words Model For the Test Portion of the Dataset (Code taken from previous assignments)
vocab = set()
bowTest = []
for text in processedSentencesTest:
    wordCounts = {}
    tokens = nltk.word_tokenize(text)
    vocab.update(tokens)

    for word in tokens:
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1

    bowTest.append(wordCounts)

# Converting Both BOW Models into a Matrix - For Multinomial Naive Bayes BOW Baseline Calculation
# Referenced https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html for implementation
dictVectorizer = DictVectorizer()
bowTrainMatrix = dictVectorizer.fit_transform(bowTrain)
bowTestMatrix = dictVectorizer.transform(bowTest)

# Training the Multinomial Naive Bayes Classifier
mnbClassifier = MultinomialNB()
mnbClassifier.fit(bowTrainMatrix, labelsTrain)

# Predicting the Test Portion of the Dataset Using the Classifier
mnbPredictions = mnbClassifier.predict(bowTestMatrix)

# Calculating the Accuracy of the Multinomial Naive Bayes Classifier
accuratePredictions = sum(mnbPredictions == actualLabels for mnbPredictions, actualLabels in zip(mnbPredictions, labelsTest))
accuracy = accuratePredictions / len(labelsTest)
print("\nMultinomial Naive Bayes Baseline Accuracy: ", accuracy)

# Random Baseline
randomPredictions = [random.randint(0, 1) for i in range(len(labelsTest))]
accurateRandomPredictions = sum(randomPredictions == actualLabels for randomPredictions, actualLabels in zip(randomPredictions, labelsTest))
randomAccuracy = accurateRandomPredictions / len(labelsTest)
print("Random Baseline Accuracy: ", randomAccuracy)

# Target-Class Baseline
targetClass = 1  # Where 1 represents spam messages
targetPredictions = [targetClass for i in range(len(labelsTest))]
accurateTargetPredictions = sum(targetPredictions == actualLabels for targetPredictions, actualLabels in zip(targetPredictions, labelsTest))
targetAccuracy = accurateTargetPredictions / len(labelsTest)
print("Target-Class Baseline Accuracy: ", targetAccuracy)
