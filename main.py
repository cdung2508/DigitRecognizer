from tkinter import *
from tkinter import messagebox
import PIL.Image
import numpy as np
import cv2
from PIL import Image, EpsImagePlugin
from sklearn.preprocessing import StandardScaler
import joblib
EpsImagePlugin.gs_windows_binary = r'venv/Include/gs/gs9.21/bin/gswin64c'


class App(Tk):
    def __init__(self, theta=None, model=None):
        super().__init__()
        # self.model = model
        self.theta = theta
        self.model = model
        self.y_predict = 0
        self.probability = 0

        self.title("Digit Recognize")
        self.resizable(width=False, height=False)
        self.geometry('300x360')
        self.config(bg='seashell2')

        # canvas
        self.c = Canvas(self, bg='White', width=230, height=226)
        # self.c = Canvas(self, bg='White', width=200, height=196)
        self.c.pack(pady=3)
        self.c.bind('<B1-Motion>', self.draw)

        # button
        self.btn_predict = Button(self, text="Predict", font=("Segoe UI Bold", 15), width=8, height=1,
                                  border=0, command=self.predict)
        self.btn_predict['bd'] = 2
        self.btn_predict['bg'] = 'seashell3'
        self.btn_predict.place(x=30, y=305)

        self.btn_clear = Button(self, text="Clear", font=("Segoe UI Bold", 15), width=8, height=1, border=0,
                                command=self.clear)
        self.btn_clear['bg'] = 'seashell3'
        self.btn_clear['bd'] = 2
        self.btn_clear.place(x=155, y=305)

        # label to display result
        self.lbl_predict = Label(self, text="Number predict:\t", font=("Segoe UI Bold", 12), bg='White')
        self.lbl_predict.place(x=65, y=237)

        self.lbl_proba = Label(self, text='Probability:\t', font=("Segoe UI Bold", 12), bg='White')
        self.lbl_proba.place(x=65, y=270)

        # show message
        messagebox.showinfo("Note", """Draw the number in the center of the frame to get 
the most probability of correct prediction !!!""")

    def draw(self, event):
        x1, y1 = event.x - 1, event.y - 1
        x2, y2 = event.x + 1, event.y + 1
        self.c.create_oval(x1, y1, x2, y2, fill="#000000", width=12)

    def predict(self):
        self.convert_to_png()
        self.img_processing()
        data = self.convert_image_to_array("./save_image/convert.png").reshape(1, -1)
        if np.all(data == np.zeros((1, 784), dtype=np.float64)):
            return
        data_with_bias = np.c_[np.ones((1, 1)), data]

        if self.theta is not None:
            logits = data_with_bias.dot(self.theta)
            y_proba = self.softmax(logits)
            self.y_predict = np.argmax(y_proba)
            self.probability = np.round(y_proba[0][self.y_predict] * 100, 2)
        elif str(self.model) == 'KNeighborsClassifier(n_neighbors=3)':
            self.y_predict = self.model.predict(data)[0]
            self.probability = np.round(self.model.predict_proba(data)[0][self.y_predict] * 100 - 5, 2)
        else:
            self.y_predict = self.model.predict(data)[0]
            self.probability = np.round(self.model.predict_proba(data)[0][self.y_predict] * 100, 2)

        self.lbl_predict.config(text=f"Number predict: {self.y_predict}")
        self.lbl_proba.config(text=f'Probability: {self.probability - 5}%')

    def clear(self):
        self.c.delete('all')
        self.lbl_predict.config(text="Number predict:\t")
        self.lbl_proba.config(text="Probability:\t")

    def convert_to_png(self):
        self.c.update()
        self.c.postscript(file='./save_image/number.ps')
        img = Image.open('./save_image/number.ps')
        img.save('./save_image/convert.png')

    @staticmethod
    def img_processing():
        image = cv2.imread('./save_image/convert.png')
        dim = (28, 28)
        resized_img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite('./save_image/convert.png', resized_img)

    @staticmethod
    def convert_image_to_array(path):
        import matplotlib.pyplot as plt
        image = PIL.Image.open(path)
        # noinspection PyTypeChecker
        image_array = np.array(image)[:, :, 0]
        image_array = np.array([255 - i for i in image_array.reshape(784, )]).reshape(1, -1)
        std = joblib.load('model/std.pkl')
        image_array = std.transform(image_array)
        return image_array

    @staticmethod
    def softmax(logits):
        exps = np.exp(logits)
        return exps / np.sum(exps, axis=1, keepdims=True)


if __name__ == '__main__':
    # knn_clf = joblib.load('model/knn_clf.pkl')
    # log_reg = joblib.load('model/log_reg.pkl')
    svm_clf = joblib.load('model/svm_clf.pkl')
    # theta_et = []
    # with open('model/weights_mgd.txt', 'r') as f:
    #     weights = [float(i) for i in f.read().split()]
    # for i in range(0, len(weights), 10):
    #     theta_et.append(weights[i:i + 10])
    app = App(model=svm_clf)
    app.mainloop()
