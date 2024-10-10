import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets

def sign(z):
    if z >= 0:
        return 1
    else:
        return -1

class Perceptron():
    def __init__(self):
        self.filename = ''
        self.epoch = 100
        self.lr = 0.5
        self.data = []
        self.x_num = []
        self.fin_weights = []
        self.train_data = []
        self.test_data = []
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
    

    def readfile(self):
        self.filename = input()
        file = open('basic/' + self.filename, "r")

        for line in file.readlines():
            line = line.rstrip('\n')
            line = '-1 ' + line
            line = line.split(' ')
            line = np.float_(line)
            # print(type(line))
            # print(line)
            self.data.append(line)

        file.close()
        # print(data)

    def preprocess(self):
        self.x_num = len(self.data[0]) - 1
        class1 = self.data[0][-1]
        for i in self.data:
            if i[-1] == class1:
                i[-1] = -1
            else:
                i[-1] = 1

        train_data, test_data = train_test_split(self.data, test_size=0.33)
        self.train_data = np.array(train_data)
        self.test_data = np.array(test_data)
        # print(type(train_data))

    def train(self):
        w = np.random.uniform(0, 1, self.x_num)

        for epoch in range(self.epoch):
            for i in self.train_data:
                y = np.dot(w, i[:self.x_num])
                d = sign(y)
                if y >= 0 and i[-1] != d:
                    w = w - self.lr * i[:-1]
                elif y < 0 and i[-1]!= d:
                    w = w + self.lr * i[:-1]
            print("epoch: ", epoch, "  weights: ", w)

        self.fin_weights.append(w)
        # print(self.fin_weights)

    def predict(self):
        train_predictions = []
        train_classes = []
        weights = np.array(self.fin_weights)
        for i in self.train_data:
            y = np.dot(weights, i[:self.x_num])
            d = sign(y)
            train_predictions.append(d)
            train_classes.append(i[-1])
        
        test_predictions = []
        test_classes = []
        for i in self.test_data:
            y = np.dot(weights, i[:self.x_num])
            d = sign(y)
            test_predictions.append(d)
            test_classes.append(i[-1])
        
        self.train_accuracy = accuracy_score(train_classes, train_predictions)
        self.test_accuracy = accuracy_score(test_classes, test_predictions)
        print("Train Accuracy: ", self.train_accuracy)
        print("Test Accuracy: ", self.test_accuracy)




if __name__ == "__main__":
    perceptron = Perceptron()
    perceptron.readfile()
    perceptron.preprocess()
    perceptron.train()
    perceptron.predict()