# Importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# to compare our model's accuracy with sklearn model
from sklearn.linear_model import LogisticRegression


# Logistic Regression
class CustomLogisticRegression:
    def __init__(self, learning_rate: float, iterations: int):
        self.learning_rate = learning_rate
        self.iterations = iterations

        self.num_of_training_examples = None
        self.num_of_features = None
        self.W, self.b, self.X, self.Y = None, None, None, None

    # Function for model training
    def fit(self, X, Y):
        # no_of_training_examples, no_of_features
        self.num_of_training_examples, self.num_of_features = X.shape

        # weight initialization
        self.W = np.zeros(self.num_of_features)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent learning
        for i in range(self.iterations):
            self.update_weights()
        return self

    @staticmethod
    def sigmoid(X, W, b):
        output = 1 / ( 1 + np.exp(-(X.dot(W)+b)))
        return output

    # Helper function to update weights in gradient descent
    def update_weights(self):
        A = self.sigmoid(self.X, self.W, self.b)

        # calculate gradients
        difference = (A - self.Y.T)
        difference = np.reshape(difference, self.num_of_training_examples)
        dW = np.dot(self.X.T, difference) / self.num_of_training_examples
        db = np.sum(difference) / self.num_of_training_examples

        # update weights
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db

        return self

    # Hypothetical function h( x )
    def predict(self, X, threshold: float = 0.5):
        Z = self.sigmoid(X, self.W, self.b)       
        Y = np.where( Z > threshold,1,0 )  
        return Y