"""Perceptron Learning Algorithm

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as _sigmoid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


np.random.seed(0)

def gen_samples(num_samples: int = 1000, num_features: int = 30, ):
    X, y = make_classification(n_samples=num_samples, n_features=num_features, n_redundant=0, n_classes=2, flip_y=0, 
                               n_clusters_per_class=1, class_sep=3, weights=[0.85, 0.15])    
    y = np.where(y > 0, 1., -1)
    return X, y


def sign(x):
    return np.where(x > 0, 1, -1)


def predict(X, w, b):
    return sign(X @ w + b)


class PerceptronClassification:

    def __init__(self, learning_rate: float = 0.001, num_iters: int = 100, batch_size: int = 100):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.batch_size = batch_size
    
    def fit(self, X, y, valid_X=None, valid_y=None):
        w = self.w = np.ones(X.shape[1])
        b = self.b = np.zeros(1)
        learning_rate = self.learning_rate
        
        for iter_id in range(self.num_iters):
            running_loss = 0.
            n_samples = 0.
            for epoch, (data, label) in enumerate(zip(X, y)):
                
                s = data @ w + b
                if -label * s >= 0:                    
                    dw = -label * data
                    db = -label
                    
                    w -= learning_rate * dw
                    b -= learning_rate * db
                    running_loss += -label * s
                n_samples += 1
                
                if epoch % 200 == 0:
                    print(f'{iter_id = }/{epoch = }. running loss: {running_loss / n_samples}.')

            train_error = np.mean((self.predict(X) != y))
            if valid_X is not None and valid_y is not None:
                valid_error = np.mean((self.predict(valid_X) != valid_y))
                print(f'{iter_id = }, train error: {train_error}, valid error: {valid_error}, {w = }, {b = }')    
            else:
                print(f'{iter_id = }, train error: {train_error}, {w = }, {b = }')    
            
    def predict(self, X):
        return predict(X, self.w, self.b)


def main():
    num_samples = 1000
    num_features = 2
    X, y = gen_samples(num_samples=num_samples, num_features=num_features)
    print(f'X: {X.shape}, y: {y.shape}')
    x1_test, x2_test = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    x_test = np.array([x1_test, x2_test]).reshape(2, -1).T

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.8)
    model = PerceptronClassification(learning_rate=0.001, num_iters=15, batch_size=1)
    model.fit(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    
    max_range = np.max(train_X)
    plt.figure()
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y)
    plt.xlim(-1.2 * max_range, 1.2*max_range)
    plt.ylim(-1.2 * max_range, 1.2*max_range)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == '__main__':
    main()