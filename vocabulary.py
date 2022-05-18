import underthesea
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from underthesea.pipeline import word_tokenize
import matplotlib.pyplot as plt

url = "/home/mvhung12c1/nlp_project/data/Data_comment.xlsx"
data = pd.read_excel(url)
dataset = np.array(data)
labels = dataset[:,0]
review = dataset[:,1]
review = list(np.squeeze(review))
stopword=pd.read_csv("/home/mvhung12c1/nlp_project/data/vietnamese-stopwords-dash.txt",header=None,names=['Stopwords'])
stopword = list(stopword['Stopwords'])

def vn_tokenizer(sequence_list) -> list:
    toked_sequence = list(map(lambda x: underthesea.word_tokenize(x), sequence_list))
    return toked_sequence

def remove_stopword(sequence, stopword_list) -> list:
    for word in sequence:
        if word in stopword:
            sequence.remove(word)
    return sequence

def create_vocabulary(sequence_list) -> dict:
    vocabulary = {}
    i = 1
    for sequence in sequence_list:
        for word in sequence:
            if word not in vocabulary:
                vocabulary[word] = i
                i += 1
    return vocabulary

toked_sequence = vn_tokenizer(review)
sequence_without_sw = remove_stopword(toked_sequence, stopword)
vocab = create_vocabulary(sequence_without_sw)