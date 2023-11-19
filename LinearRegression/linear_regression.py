import numpy as np

class LinearRegression:
    
    def __init__(self, learning_rate = 0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X,y):
        #init params
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        #why should number of weights be n_Features and not n_samples?
        self.bias = 0
        # run loop till n_iterations
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X,self.weights)+ self.bias
            dw = (1/n_samples)*np.dot(X.T,(y_predicted - y)) #derivative of cost function with respect to weight turns out to be this (exlcuding 2 in the formula which is scaling factor)
            db = (1/n_samples) * np.sum(y_predicted-y) #derivative of cost function with respect to bias

            #adjusting parameters according to their derivatives
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            

    def predict(self,X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
