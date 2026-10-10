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
loss = 0.5 * (error ** 2)

# Display Results
print("===== FORWARD PROPAGATION =====")
print("Hidden Net Input (z1):", z1)
print("Hidden Output (h1):", h1)

print("\nHidden Net Input (z2):", z2)
print("Hidden Output (h2):", h2)

print("\nHidden Net Input (z3):", z3)
print("Hidden Output (h3):", h3)

print("\nOutput Net Input (z4):", z4)
print("Predicted Output (y1):", y1)

print("\nActual Output (t1):", t1)
print("Error:", error)
print("Loss:", loss)

