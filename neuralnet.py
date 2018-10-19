# Daniel Peek qer419
# Michael Canas ohh135
# CS3793 Assignment 04
# 10/19/2018
# neuralnet.py based on Dr. O'Hara's bp.c
# python 3


import random
import math


# create random starting weights
def rand():
    number = random.uniform(-1, 1)
    return number


# sigmoid logistic function
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))


# derivative of sigmoid
def derivative_sigmoid(x):
    return x * (1.0 - x)


# class for the neural net
class NeuralNetwork:
    def __init__(self, number_inputs, number_hidden, number_outputs, eta):
        self.inputs = number_inputs
        self.hidden_nodes = number_hidden
        self.classes = number_outputs

        self.eta = eta
        self.confidence = 0

        # weights
        self.bias_bottom = [rand() for i in range(self.hidden_nodes)]
        self.bias_top = [rand() for i in range(self.classes)]
        self.weight_bottom = [[rand() for i in range(self.hidden_nodes)] for j in range(self.inputs)]
        self.weight_top = [[rand() for i in range(self.classes)] for j in range(self.hidden_nodes)]

        # hidden and output nodes
        self.hidden = [0.0] * self.hidden_nodes
        self.output = [0.0] * self.classes

    # forward prediction
    def predict(self, sample):
        i = 0
        # propagate sample and weights to hidden nodes
        for k in range(self.hidden_nodes):
            weight_sum = 0.0
            for i in range(self.inputs):
                weight_sum += self.weight_bottom[i][k] * sample[i]
            weight_sum += self.bias_bottom[k]
            self.hidden[k] = sigmoid(weight_sum)

        # propagate hidden nodes to output
        for k in range(self.classes):
            weight_sum = 0.0
            for i in range(self.hidden_nodes):
                weight_sum += self.weight_top[i][k] * self.hidden[i]
            weight_sum += self.bias_top[k]
            self.output[k] = sigmoid(weight_sum)

        # choose to stay (0) or draw (1), by taking the higher output
        for k in range(self.classes):
            if k == 0 or self.output[k] > self.output[i]:
                i = k

        # find confidence by seeing how preferred the chosen value was
        next_max_sum = -1.0
        for k in range(self.classes):
            if k != i:
                if next_max_sum < 0.0 or self.output[k] > next_max_sum:
                    next_max_sum = self.output[k]
        self.confidence = self.output[i] - next_max_sum

        # return stay (0) or draw (1)
        return i

    # backward propagation
    def adjust_weights(self, sample, actual):
        delta = [0] * self.classes
        # correct outputs nudge top weights in that direction
        for k in range(self.classes):
            if k == actual:
                weight_sum = 1.0
            else:
                weight_sum = 0.0
            weight_sum -= self.output[k]

            delta[k] = weight_sum * derivative_sigmoid(self.output[k])

            for i in range(self.hidden_nodes):
                self.weight_top[i][k] += self.eta * delta[k] * self.hidden[i]

            self.bias_top[k] += self.eta * delta[k]

        # use hidden nodes and top weights to adjust bottom weights in the right direction
        for j in range(self.hidden_nodes):
            d = 0.0
            for k in range(self.classes):
                d += self.weight_top[j][k] * delta[k]

            for i in range(self.inputs):
                self.weight_bottom[i][j] += self.eta * derivative_sigmoid(self.hidden[j]) * d * sample[i]

            self.bias_bottom[j] += self.eta * d
