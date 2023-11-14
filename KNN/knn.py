import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k=3):
        self.k = k
    
    def fit(self, x,y):
        self.X_train = x
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict_sample(x) for x in X]
        return np.array(predicted_labels)


    def _predict_sample(self,x):
        #Compute distances from each sample
        #Get K nearest samples, labels
        #majority vote, most common class label
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train] #calculating distance from current sample to all other samples

        k_indices = np.argsort(distances)[:self.k] #sorting the distances and getting the indices of nearest labels uptil K (here, 3 so 3 nearest neighbours)
        k_nearest_labels = [self.y_train[i] for i in k_indices] #getting the nearest labels

        #majority vote here
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common
    