# ==========================================================
# 3-3-2 ANN  (3 inputs -> 3 hidden neurons -> 2 output)
# Written in the same style as the FP-BP lab PDF
# ==========================================================


# ----------------------------------------------------------
# fp
# ----------------------------------------------------------
import numpy as np

# Step 1: Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Step 2: Inputs
x1 = 0.6
x2 = 0.8
x3 = 0.5

# Step 3: Hidden Layer Weights and Bias
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
w7 = 0.6
w8 = 0.3
w9 = 0.5
b3= 0.1
# Step 4: Hidden Neuron Calculation
z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1
h1 = sigmoid(z1)

z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2
h2 = sigmoid(z2)

z3  = (x1 * w7) + (x2 * w8) +(x3 * w9)+ b3
h3 =  sigmoid(z3)

# Step 5: Output Layer Weights and Bias
#N1
w10 = 0.9
w11 = 0.8
w12 = 0.7
b4 = 0.1

#N2
w13 = 0.5
w14 = 0.6
w15 = 0.4
b5 = 0.2

# Step 6: Output Neuron Calculation
z4 = (h1 * w10) + (h2 * w11) + (h3*w12) + b4
y1 = sigmoid(z4)
z5 = (h1 * w13) + (h2 * w14) + (h3*w15) + b5
y2 = sigmoid (z5)
# Step 7: Actual Output
t1 = 1
t2 = 0
# Step 8: Error and Loss
error1 = t1 - y1
error2 = t2 - y2
loss = 0.5 * (error1 ** 2) + 0.5 * (error2 ** 2)


