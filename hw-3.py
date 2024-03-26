from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def tokenizeBert(text):
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)


def tokenizeT5(text):
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)



def main():
    dataset = load_dataset('sms_spam')

    train = dataset['train']['sms'][0:4456]
    test = dataset['train']['sms'][0:1114]

    tokenizedBert = train.map(tokenizeBert, batched=True)  # https://huggingface.co/google-bert/bert-base-cased
    tokenizedT5 = train.map(tokenizeT5, batched=True)  # https://huggingface.co/google/flan-t5-base

    modelBert = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
    modelT5 = AutoModelForSequenceClassification.from_pretrained("google/flan-t5-base", num_labels=2)

    #Left off on the Training hyperparameters section of the website

if __name__ == '__main__':
    main()
