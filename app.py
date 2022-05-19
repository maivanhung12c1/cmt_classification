import tornado
import tornado.web
import tornado.ioloop
import tornado.template
from tensorflow import keras
import pandas as pd
from preprocess.preprocess import Preprocess


class MainHandler(tornado.web.RequestHandler):

    def initialize(self):
        self.model = keras.models.load_model("model")
        self.label_names = ['tào lao', 'ngon', 'dở', 'giá cao', 'giá hợp lý', 
               'vệ sinh sạch sẽ, thực phẩm an toàn', 'vệ sinh bẩn, không đảm bảo', 
               'thái độ phục vụ tốt', 'thái độ phục vụ tệ']
        self.preprocess = Preprocess()

    def get(self):
        return self.render("templates/home.html", predict="")

    def classify(self, text):
        padded_sequence = self.preprocess.preprocess_for_text(text)
        pred = self.model.predict(padded_sequence)
        classes = [self.label_names[i] for i in list(pred.argmax(axis=1))]
        return classes

    def post(self):
        if self.get_argument("comment") is not None:
            input_cmt = []
            input_cmt.append(self.get_argument("comment"))          
            predict = self.classify(input_cmt)[0]
            return self.render("templates/home.html", predict=predict)

def main():
    app = tornado.web.Application([
        (r"/", MainHandler),
    ])
    port = 9999
    app.listen(port)
    print("Running")
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()