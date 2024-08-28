""" Logistic regression from scratch
"""
import numpy as np
from scipy.special import expit as _sigmoid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

def gen_samples(num_samples: int = 1000, num_features: int = 30, ):
    return make_classification(n_samples=num_samples, n_features=num_features, n_classes=2, weights=[0.85, 0.15])    


def gen_data(X, y, batch_size: int):
    
    n = X.shape[0]
    for i in range(0, n, batch_size):
        data = X[i:i+batch_size]
        lables = y[i:i+batch_size]
        yield data, lables

def sigmoid(X, w, b):
    # print(f'{X.shape = }, {w.shape = }, {b = }')
    
    z = X @ w + b
    return _sigmoid(z)
    

class LogisticRegression:
    
    def __init__(self, learning_rate: float = 0.001, num_iters: int = 100, batch_size: int = 100):
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.batch_size = batch_size
    
    def fit(self, X, y, valid_X=None, valid_y=None):
        weights = np.zeros(X.shape[1])
        bias = np.zeros(1)
        
        for iter_id in range(self.num_iters):
            # self.batch_size = 1
            running_loss = 0.
            n_samples = 0
            for epoch, (data, labels) in enumerate(gen_data(X, y, batch_size=self.batch_size)):
                m = len(data)
                s = sigmoid(data, weights, bias)
                # print(f'{labels = }, {s = }')
                loss = log_loss(labels, s)
                running_loss += loss * m
                n_samples += m
                delta = s - labels
                # print(f'{X.shape = }, {delta.shape = }, {m = }')
                dw = data.T @ delta  / m
                db = np.sum(delta) / m
                weights -= self.learning_rate * dw
                bias -= self.learning_rate * db
                if epoch % 50 == 0:
                    if valid_X is not None and valid_y is not None:
                        valid_s = sigmoid(valid_X, weights, bias)
                        vlaid_loss = log_loss(valid_y, valid_s)
                        
                        print(f'[{iter_id:02d}/{epoch:02d}] train loss: {loss}, train running loss: {running_loss / n_samples:04f}, '
                            f'valid loss: {vlaid_loss}')

                    else:
                        print(f'[{iter_id:02d}/{epoch:02d}] train loss: {loss}, train running loss{running_loss / n_samples:04f}')

                    
        self.weights = weights
        self.bias = bias
    
    def predict_proba(self, X):
        return sigmoid(X, self.weights, self.bias)

        

def main():
    num_samples = 10000
    num_features = 30
    X, y = gen_samples(num_samples=num_samples, num_features=num_features)
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, train_size=0.8)
    print(f'#{train_X.shape = }, {valid_X.shape = }')
    model = LogisticRegression(num_iters=10, batch_size=256)
    model.fit(train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    predicts = model.predict_proba(valid_X)
    valid_loss = log_loss(valid_y, predicts)
    print(f'valid loss = {valid_loss}')
    
    

if __name__ == '__main__':
    main()