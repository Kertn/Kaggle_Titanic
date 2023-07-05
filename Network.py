import random
import numpy
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(75)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

class Neurons:
    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.w = [np.random.randn(y, x)  for x, y in zip(sizes[:-1], sizes[1:])]
        self.delta_list = []
        self.eror_list = []

    def forward_pass(self, input, y):
        input = input.astype(float)
        a_list = [input]
        sum_list = []
        a = input
        for w, b in zip(self.w[:-1], self.biases[:-1]):
            a = np.reshape(a, (len(a), 1))
            sum_list.append(np.dot(w, a) + b)
            a = sigmoid(np.dot(w, a) + b)
            a_list.append(a)

        sum_list.append(np.dot(self.w[-1], a) + self.biases[-1])
        y_hat = sigmoid(sum_list[-1])
        print('forward_pass', y_hat)
        a_list.append(y_hat)
        delta = (y_hat - y) * sigmoid_prime(sum_list[-1])
        self.eror_list.append(y_hat[0][0] - y)
        print("eror_list", y_hat[0][0] - y)
        self.delta_list.append(delta[0][0])
        return delta[0][0], a_list, sum_list

    def backprop(self, input, y):
        delta, a_list, sum_list = self.forward_pass(input, y)
        nabla_w = [np.zeros(b.shape) for b in self.w]
        nabla_b = [np.zeros(a.shape) for a in self.biases]
        nabla_w[-1] = np.dot(delta, np.reshape(a_list[-2], (1, len(a_list[-2]))))
        nabla_b[-1] = np.array(delta)
        for layer in range(2, self.num_layer):
            delta = (np.dot(self.w[-layer+1].transpose(), delta) * sigmoid_prime(sum_list[-layer]))
            act = np.reshape(a_list[-layer-1], (len(a_list[-layer-1]), 1))
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, act.transpose())
        return nabla_w, nabla_b

    def update(self, batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.w]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for input, y in batch:
            nabla_new_w, nabla_new_b = self.backprop(input, y)
            nabla_w = [nw + nnw for nw, nnw in zip(nabla_w, nabla_new_w)]
            nabla_b = [nb + nnb for nb, nnb in zip(nabla_b, nabla_new_b)]
        eps = eta / len(batch)
        self.biases = [b - eps * nabla_b for b, nabla_b in zip(self.biases, nabla_b)]
        self.w = [w - eps * nabla_w for w, nabla_w in zip(self.w, nabla_w)]


    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            batch_data = [training_data[k:k + mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for batch in batch_data:
                self.update(batch, eta)
        if test_data is not None:
            for test in test_data:
                self.predict(test[0])

    def predict(self, input):
        input = input.astype(float)
        a = input
        for w, b in zip(self.w[:-1], self.biases[:-1]):
            a = np.reshape(a, (len(a), 1))
            a = sigmoid(np.dot(w, a) + b)

        y_hat = sigmoid(np.dot(self.w[-1], a) + self.biases[-1])
        print("predict", y_hat)

    def make_graph(self):
        #print(self.eror_list)
        # print(self.delta_list)
        # print(len(self.eror_list))
        end = [sum(self.eror_list[i-100:i])/100 for i in range(100, len(self.eror_list), 100)]
        plt.plot(end)
        plt.grid()
        plt.show()
