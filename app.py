import tornado
import tornado.web
import tornado.ioloop
import tornado.template
import tensorflow
from tensorflow import keras
import underthesea
import tensorflow as tf
import numpy as np
import pandas as pd
from underthesea.pipeline import word_tokenize
import matplotlib.pyplot as plt
import vocabulary


class MainHandler(tornado.web.RequestHandler):

    def initialize(self):
        #super(MainHandler, self).__initialize__()
        url="/home/mvhung12c1/nlp_project/data/vietnamese-stopwords-dash.txt"
        stopword=pd.read_csv(url,header=None,names=['Stopwords'])
        self.stopword = list(stopword['Stopwords'])
        self.vocabulary = vocabulary.vocab
        self.model = keras.models.load_model("/home/mvhung12c1/nlp_project/model")
        self.label_names = ['tào lao', 'ngon', 'dở', 'giá cao', 'giá hợp lý', 
               'vệ sinh sạch sẽ, thực phẩm an toàn', 'vệ sinh bẩn, không đảm bảo', 
               'thái độ phục vụ tốt', 'thái độ phục vụ tệ']

    #def stopword(self, url="/home/mvhung12c1/nlp_project/data/vietnamese-stopwords-dash.txt"):
    #    stopword=pd.read_csv(url,header=None,names=['Stopwords'])
        #stopword = list(stopword['Stopwords'])
    #    return stopword

    def vn_tokenizer(self, sequence_list) -> list:
        toked_sequence = list(map(lambda x: underthesea.word_tokenize(x), sequence_list))
        return toked_sequence

    def remove_stopword(self, sequence, stopword_list) -> list:
        stopword = self.stopword
        for word in sequence:
            if word in stopword:
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

    def get(self):
        return self.render("templates/home.html", predict="")

    def classify(self, text):
        #stopword = list(self.stopword['Stopwords'])
        toked_sequence = self.vn_tokenizer(text)
        sequence_without_sw = self.remove_stopword(toked_sequence, self.stopword)
        num_sequence = self.map_word_to_num(sequence_without_sw, self.vocabulary)
        max_length = self.find_max_length(num_sequence)
        padded_sequence = self.sequence_padding(num_sequence, max_length, 0)
        padded_sequence = np.array(padded_sequence).astype("float32")
        pred = self.model.predict(padded_sequence)
        classes = [self.label_names[i] for i in list(pred.argmax(axis=1))]
        return classes

    def post(self):
        if self.get_argument("comment") is not None:
            input_cmt = self.get_argument("comment")           
            predict = self.classify(input_cmt)
            return self.render("templates/home.html", predict=predict)


def main():
    app = tornado.web.Application([
        (r"/", MainHandler),
    ])
    port = 8080
    app.listen(port)
    print("Running")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()