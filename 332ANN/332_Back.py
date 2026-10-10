import numpy as np
# Sigmoid Function
def sigmoid(x):
 return 1 / (1 + np.exp(-x))
# Sigmoid Derivative
def sigmoid_derivative(x):
 return x * (1 - x)
# Inputs
x1 = 0.6
x2 = 0.8
x3 = 0.5
# Initial Weights and Biases
# Hidden neuron 1
w1 = 0.4
w2 = 0.7
w3 = 0.3
b1 = 0.2
# Hidden neuron 2
w4 = 0.5
w5 = 0.2
w6 = 0.6
b2 = 0.1
# Hidden neuron 3
w7 = 0.3
w8 = 0.6
w9 = 0.4
b3 = 0.3
# Output neuron 1
w10 = 0.9
w11 = 0.8
w12 = 0.5
b4 = 0.1
# Output neuron 2
w13 = 0.3
w14 = 0.7
w15 = 0.6
b5 = 0.2
# Learning Rate
lr = 0.1
# Target Output
t1 = 1
t2 = 0
# ====================
# FORWARD PROPAGATION
# ====================
z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1
h1 = sigmoid(z1)
z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2
h2 = sigmoid(z2)
z3 = (x1 * w7) + (x2 * w8) + (x3 * w9) + b3
h3 = sigmoid(z3)
z4 = (h1 * w10) + (h2 * w11) + (h3 * w12) + b4
y1 = sigmoid(z4)
z5 = (h1 * w13) + (h2 * w14) + (h3 * w15) + b5
y2 = sigmoid(z5)
# Error
error1 = t1 - y1
error2 = t2 - y2

#Backpropagation

delta_y1 = error1 * sigmoid_derivative(y1)
delta_y2 = error2 * sigmoid_derivative(y2)

 
delta_h1 = (delta_y1 * w10 + delta_y2 * w13) *sigmoid_derivative(h1)
delta_h2 = (delta_y1 * w11 + delta_y2 * w14) * sigmoid_derivative(h2)
delta_h3 = (delta_y1 * w12 + delta_y2 * w15) * sigmoid_derivative(h3)

 
# ====================
# WEIGHT UPDATES
# ====================

w10 = w10 + (lr * h1 * delta_y1)
w11 = w11 + (lr * h2 * delta_y1)
w12 = w12 + (lr * h3 * delta_y1)
b4 = b4 + (lr * delta_y1)
w13 = w13 + (lr * h1 * delta_y2)
w14 = w14 + (lr * h2 * delta_y2)
w15 = w15 + (lr * h3 * delta_y2)
b5 = b5 + (lr * delta_y2)
w1 = w1 + (lr * x1 * delta_h1)
w2 = w2 + (lr * x2 * delta_h1)
w3 = w3 + (lr * x3 * delta_h1)
b1 = b1 + (lr * delta_h1)
w4 = w4 + (lr * x1 * delta_h2)
w5 = w5 + (lr * x2 * delta_h2)
w6 = w6 + (lr * x3 * delta_h2)
b2 = b2 + (lr * delta_h2)
w7 = w7 + (lr *  x1 * delta_h3)
w8 = w8 + (lr * x2 * delta_h3)
w9 = w9 + (lr * x3 * delta_h3)
b3 = b3 + (lr * delta_h3)

 
