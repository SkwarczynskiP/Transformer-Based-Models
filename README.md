# Transformer-Based-Models
 CSI 4180 - Natural Language Processing - Homework 3

This Transformer-Based Models project:
 - Utilizes the HuggingFace SMS Spam Detection dataset (https://huggingface.co/datasets/sms_spam)
 - Divides the dataset into Training and Testing sets
 - Fine-tunes the prajjwal1/bert-tiny and the distilroberta-base models on the dataset
 - Evaluates the accuracy of running the Fine-tuned models on the Testing set
 - Runs Zero-shot prompt classification of the mistralai/Mistral-7B-Instruct-v0.1 and bigscience/bloomz-560m models on the Testing set
 - Generates Multinomial Naive Bayes BOW, Random, and Target-class baselines on the Testing set for comparison with the Transformer-Based models

# Files:
The files in this repository include:
 - `hw-3.py`: The Python script containing the code
 - `Section2.3.ipynb`: The Jupyter Notebook containing the code for section 2.3
 - `Report-hw3.pdf`: The PDF report containing analysis of the results

# How to Run:
1. Download the files
2. Open the terminal and change directories to the location of the downloaded files
3. Type the command "py hw-3.py" to run

Note: Section 2.3 is not a part of the main script and must be run separately in a Jupyter Notebook
