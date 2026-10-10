# ==========================================================
# 3-3-1 ANN  (3 inputs -> 3 hidden neurons -> 1 output)
# Written in the same style as the FP-BP lab PDF
# ==========================================================


# ----------------------------------------------------------
# fp
# ----------------------------------------------------------
import numpy as np

# Step 1: Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x* (1-x)


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
w9 = 0.6
w10 = 0.3
w11 = 0.5
b4= 0.1
# Step 4: Hidden Neuron Calculation
z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1
h1 = sigmoid(z1)

z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2
h2 = sigmoid(z2)

z3  = (x1 * w9) + (x2 * w10) +(x3 * w11)+ b4
h3 =  sigmoid(z3)

# Step 5: Output Layer Weights and Bias
w7 = 0.9
w8 = 0.8
w12 = 0.7
b3 = 0.1

# Step 6: Output Neuron Calculation
z4 = (h1 * w7) + (h2 * w8) + (h3*w12) + b3
y1 = sigmoid(z4)

# Step 7: Actual Output
t1 = 1

# Step 8: Error and Loss
error = t1 - y1

#Backpropagation

delta_y1 = error * sigmoid_derivative(y1)
 
delta_h1 = delta_y1 * w7 * sigmoid_derivative(h1)
delta_h2 = delta_y1 * w8 * sigmoid_derivative(h2)
 
# ====================
# WEIGHT UPDATES
# ====================
 
w7 = w7 + (lr * h1 * delta_y1)
w8 = w8 + (lr * h2 * delta_y1)
b3 = b3 + (lr * delta_y1)
 
w1 = w1 + (lr * x1 * delta_h1)
w2 = w2 + (lr * x2 * delta_h1)
w3 = w3 + (lr * x3 * delta_h1)
b1 = b1 + (lr * delta_h1)
 
w4 = w4 + (lr * x1 * delta_h2)
w5 = w5 + (lr * x2 * delta_h2)
w6 = w6 + (lr * x3 * delta_h2)
b2 = b2 + (lr * delta_h2)
 
# Display Results
print("\n\n===== AFTER BACKPROPAGATION =====")
 
print("\nUpdated Hidden Layer (Neuron 1)")
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("b1 =", b1)
 
print("\nUpdated Hidden Layer (Neuron 2)")
print("w4 =", w4)
print("w5 =", w5)
print("w6 =", w6)
print("b2 =", b2)
 
print("\nUpdated Output Layer")
print("w7 =", w7)
print("w8 =", w8)
print("b3 =", b3)
 
print("\nPredicted Output =", y1)
print("Error =", error)
 
 