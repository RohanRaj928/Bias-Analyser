import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


## Logistic Regression setup (OvR) ##
'''
Definitions
lr:     Learning Rate

X:      Input
y_true: True value
y_pred: Predicted value
w:      weights
b:      bias
'''

lr = 0.01
epochs = 500

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Binary cross-entropy loss
def loss(y_true, y_pred):
    epsilon = 1e-9
    return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

# Forward pass
def forward_pass(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias)

# Train binary logistic regression
def train_binary(X, y, epochs, lr):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        A = forward_pass(X, weights, bias)
        dz = A - y
        dw = (1 / n_samples) * np.dot(X.T, dz)
        db = (1 / n_samples) * np.sum(dz)
        weights -= lr * dw
        bias -= lr * db

        # Optional: track loss
        losses.append(loss(y, A))

    return weights, bias, losses

# One-vs-Rest training for multiclass
def train_ovr(X, y, epochs, lr):
    classes = np.unique(y)
    weights_dict = {}
    bias_dict = {}
    losses_dict = {}

    for cls in classes:
        print(f"\nTraining classifier for class {cls} vs rest")
        y_binary = (y == cls).astype(int)
        w, b, l = train_binary(X, y_binary, epochs, lr)
        weights_dict[cls] = w
        bias_dict[cls] = b
        losses_dict[cls] = l

    return weights_dict, bias_dict, losses_dict

# OvR prediction
def predict_ovr(X, weights_dict, bias_dict):
    classes = list(weights_dict.keys())
    probs = np.zeros((X.shape[0], len(classes)))

    for i, cls in enumerate(classes):
        probs[:, i] = forward_pass(X, weights_dict[cls], bias_dict[cls])

    y_pred = np.argmax(probs, axis=1)
    return y_pred



## Importing Dataset ##
splits = {'train': 'train_df.csv', 'test': 'test_df.csv',
          'valid': 'val_df.csv'}

df_train = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/" + splits["train"])
df_valid = pd.read_csv("hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/" + splits["valid"])

## Processing Data ##
vectorizer = TfidfVectorizer(max_features=5000)

# Fit to training data and transform
X_train = vectorizer.fit_transform(df_train['text']).toarray()

# Transform validation data
X_valid = vectorizer.transform(df_valid['text']).toarray()

y_train = df_train['label'].values
y_valid = df_valid['label'].values

## Train Model ##
# Train the OvR classifiers
weights_dict, bias_dict, losses_dict = train_ovr(X_train, y_train, epochs, lr)

# Transform new text into the same feature space
new_X = vectorizer.transform(["love, like, good, amazing, excellent, great"]).toarray()

# Get the list of classes
classes = list(weights_dict.keys())

# Predict the class index
pred_index = predict_ovr(new_X, weights_dict, bias_dict)[0]
print(pred_index)


print(df_train.head())





