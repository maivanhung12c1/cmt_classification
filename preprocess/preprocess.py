import underthesea
import numpy as np
import pandas as pd
from underthesea.pipeline import word_tokenize
import matplotlib.pyplot as plt
import string
import re


class Preprocess:

    def __init__(self,text=None, data_path="data/Data_comment.xlsx", stopword_path="data/vietnamese-stopwords-dash.txt"):
        self.text = text 
        self.data = np.array(pd.read_excel(data_path))
        self.stopword = list(pd.read_csv(stopword_path,header=None,names=['Stopwords'])['Stopwords'])
        self.all_text = list(self.data[:,1])
        self.all_label = list(self.data[:,1])
        self.max_length = 0
        self.vocab = self.preprocess_and_create_vocab()

    def clean_text(self, text):
        text = text.replace(string.punctuation, "")
        text = re.sub(r'/\d+', '', text)
        text = text.replace(".", "")
        text = text.lower().split()
        text = " ".join(text)
        return text   

    def vn_tokenizer(self, sequence_list) -> list:
        toked_sequence = list(map(lambda x: underthesea.word_tokenize(x), sequence_list))
        return toked_sequence

    def remove_stopword(self, sequence) -> list:
        for word in sequence:
            if word in self.stopword:
                sequence.remove(word)
        return sequence

    def create_vocabulary(self, sequence_list) -> dict:
        vocabulary = {}
        i = 1
        for sequence in sequence_list:
            for word in sequence:
                if word not in vocabulary:
                    vocabulary[word] = i
                    i += 1
        return vocabulary
    def map_word_to_num(self, sequence_list, vocabulary) -> list:
        mapped_sequence_list = []
        for sequence in sequence_list:
            text = list(map(lambda x: vocabulary[x], sequence))
            mapped_sequence_list.append(text)
        return mapped_sequence_list

    def find_max_length(self, sequence_list) -> int:
        len_list = []
        for sequence in sequence_list:
            len_list.append(len(sequence))
        max_length = np.max(len_list)
        return max_length

    def sequence_padding(self, sequence_list, max_length, padded_value) -> list:
        for sequence in sequence_list:
            while len(sequence) < max_length:
                sequence.append(padded_value)
        return sequence_list
    
    def preprocess_and_create_vocab(self):
        review = list(map(lambda x: str(self.clean_text(x)), self.all_text))
        toked_sequence = self.vn_tokenizer(review)
        sequence_without_sw = self.remove_stopword(toked_sequence)
        self.max_length = self.find_max_length(sequence_without_sw)
        vocab = self.create_vocabulary(sequence_without_sw)
        return vocab

    def preprocess_for_text(self, text):
        text = list(map(lambda x: str(self.clean_text(x)), text))
        toked_sequence = self.vn_tokenizer(text)
        sequence_without_sw = self.remove_stopword(toked_sequence)
        num_sequence = self.map_word_to_num(sequence_without_sw, self.vocab)
        padded_sequence = self.sequence_padding(num_sequence, self.max_length, 0)
        padded_sequence = np.array(padded_sequence).astype("float32")
        self.text = padded_sequence
        return self.text

    