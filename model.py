import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

EPOCHS = 5000
LR = 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

w1 = np.array([[0.5, -0.3],
               [0.8,  0.2]])

w2 = np.array([0.1, -0.4])

dataset = np.array([
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
])

# TRAINING
for _ in range(EPOCHS):
    for row in dataset:
        x = row[:2]
        target = row[2]

        hidden = sigmoid(np.dot(w1, x))
        out = sigmoid(np.dot(w2, hidden))

        error = target - out
        d_out = error * dsigmoid(out)

        w2 += LR * d_out * hidden
        d_h = d_out * w2 * dsigmoid(hidden)
        w1 += LR * d_h.reshape(2,1) * x

# EVALUATION
X = dataset[:, :2]
y_true = dataset[:, 2]

y_pred_probs = []
for x in X:
    hidden = sigmoid(np.dot(w1, x))
    out = sigmoid(np.dot(w2, hidden))
    y_pred_probs.append(out)

y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

print("===== MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
