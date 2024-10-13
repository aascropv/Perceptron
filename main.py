import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets

def sign(z):    #activation function
    if z >= 0:
        return 1
    else:
        return -1

class Perceptron(): #perceptron
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
    

    def readfile(self): #readfile function
        # self.filename = input()
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

    def preprocess(self):   #preprocess function
        self.x_num = len(self.data[0]) - 1
        class1 = self.data[0][-1]
        for i in self.data:     #change the class of the input data
            if i[-1] == class1:
                i[-1] = -1
            else:
                i[-1] = 1

        train_data, test_data = train_test_split(self.data, test_size=0.33)
        self.train_data = np.array(train_data)
        self.test_data = np.array(test_data)
        # print(type(train_data))

    def train(self):    #training function
        w = np.random.uniform(0, 1, self.x_num)

        for epoch in range(self.epoch):
            for i in self.train_data:
                y = np.dot(w, i[:self.x_num])   #y = w * x
                d = sign(y)     #call activation function
                if y >= 0 and i[-1] != d:       #adjust the weights
                    w = w - self.lr * i[:-1]
                elif y < 0 and i[-1]!= d:
                    w = w + self.lr * i[:-1]
            print("epoch: ", epoch, "  weights: ", w)

        self.fin_weights.append(w)
        # print(self.fin_weights)

    def predict(self):      #predict the accuracy of the train data and test data
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

class PlotCanvas(FigureCanvas):     #plot train and test results
    def __init__(self, parent=None, width=12, height=4, dpi=100):
        fig, self.axes= plt.subplots(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        
    def plot(self, data, weight, title):
        self.axes.clear()
        self.axes.set_title(title)
        colors = []
        for i in data:
            if i[-1] == -1:
                colors.append('r')
            else:
                colors.append('b')
        self.axes.scatter(data[:, 1], data[:, 2], c=colors)
        self.axes.axline((0, weight[0][0] / weight[0][2]), (weight[0][0] / weight[0][1], 0), color="black")     # w0 = w1 * x1 + w2 * x2
        self.axes.set_xlabel('x1')
        self.axes.set_ylabel('x2')
        self.draw()

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MyWidget")
        self.setWindowTitle("Perceptron")
        self.resize(900, 600)
        self.ui()

    def ui(self):
        #layout
        main_layout = QtWidgets.QVBoxLayout(self)
        left_layout = QtWidgets.QVBoxLayout()
        right_layout = QtWidgets.QVBoxLayout()
        top_layout = QtWidgets.QHBoxLayout()
        bottom_layout = QtWidgets.QHBoxLayout()

        #data file
        data_label = QtWidgets.QLabel("Data:")
        left_layout.addWidget(data_label)
        self.file_choosing = QtWidgets.QComboBox(self)
        data_folder = "basic"
        for f in os.listdir(data_folder):
            self.file_choosing.addItem(f)
        left_layout.addWidget(self.file_choosing)

        #epoch
        epoch_label = QtWidgets.QLabel("Epoch: (0 ~ 10000)")
        left_layout.addWidget(epoch_label)
        self.epoch_input = QtWidgets.QSpinBox(self)
        self.epoch_input.setMinimum(0)
        self.epoch_input.setMaximum(10000)
        self.epoch_input.setValue(100)
        left_layout.addWidget(self.epoch_input)

        #learning rate
        lr_label = QtWidgets.QLabel("Learning Rate: (0.0 ~ 1.0)")
        left_layout.addWidget(lr_label)
        self.lr_input = QtWidgets.QDoubleSpinBox(self)
        self.lr_input.setMinimum(0.0)
        self.lr_input.setMaximum(1.0)
        self.lr_input.setValue(0.5)
        left_layout.addWidget(self.lr_input)

        #train button
        self.train_button = QtWidgets.QPushButton(self)
        self.train_button.setText("Train")
        self.train_button.clicked.connect(self.train_perceptron)
        left_layout.addWidget(self.train_button)

        #accuracy label & weights label
        self.train_accuracy_label = QtWidgets.QLabel(self)
        self.train_accuracy_label.setText("Training accuracy:")
        self.train_accuracy_result = QtWidgets.QLabel(self)
        self.test_accuracy_label = QtWidgets.QLabel(self)
        self.test_accuracy_label.setText("Testing accuracy:")
        self.test_accuracy_result = QtWidgets.QLabel(self)
        self.weight_label = QtWidgets.QLabel(self)
        self.weight_label.setText("Weights:")
        self.weights_result = QtWidgets.QLabel(self)
        right_layout.addWidget(self.train_accuracy_label)
        right_layout.addWidget(self.train_accuracy_result)
        right_layout.addWidget(self.test_accuracy_label)
        right_layout.addWidget(self.test_accuracy_result)
        right_layout.addWidget(self.weight_label)
        right_layout.addWidget(self.weights_result)

        #exit button
        self.exit_button = QtWidgets.QPushButton(self)
        self.exit_button.setText("Exit")
        self.exit_button.clicked.connect(self.close)
        left_layout.addWidget(self.exit_button)

        #add layouts on the top_layout
        top_layout.addLayout(left_layout)
        top_layout.addLayout(right_layout)

        #plotting canvas
        self.train_canvas = PlotCanvas(self)
        bottom_layout.addWidget(self.train_canvas)
        self.test_canvas = PlotCanvas(self)
        bottom_layout.addWidget(self.test_canvas)

        #add top_layout & bottom_layout on the main_layout
        main_layout.addLayout(top_layout)
        main_layout.addLayout(bottom_layout)

    def resultRender(self, perceptron):
        self.train_canvas.plot(perceptron.train_data, perceptron.fin_weights, "Train data")
        self.test_canvas.plot(perceptron.test_data, perceptron.fin_weights, "Test data")
        self.train_accuracy_result.setText(str(perceptron.train_accuracy))
        self.test_accuracy_result.setText(str(perceptron.test_accuracy))
        weights_string = ""
        for i in perceptron.fin_weights:
            weights_string += str(i) + " "
        self.weights_result.setText(weights_string)

    def train_perceptron(self):
        print("Start training")
        perceptron = Perceptron()
        perceptron.filename = str(self.file_choosing.currentText())
        perceptron.epoch = self.epoch_input.value()
        perceptron.lr = self.lr_input.value()
        perceptron.readfile()
        perceptron.preprocess()
        perceptron.train()
        perceptron.predict()
        self.resultRender(perceptron)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MyWidget()
    MainWindow.show()
    sys.exit(app.exec_())