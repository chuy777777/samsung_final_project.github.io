import numpy as np
from framework_ml.puromachine import Some
from framework_ml.loss_functions import MeanAbsoluteError, MeanSquareError, RootMeanSquareError

class RANSAC():
    def __init__(self, margin):
        self.margin=margin
        self.best_params=None

    def fit(self, X, y, it):
        m,n=X.shape
        indexes=np.arange(m)
        self.best_params=None
        best_count=None
        for i in range(it):
            rand_ind=np.random.choice(indexes, size=n + 1, replace=False)
            A=np.concatenate([X[rand_ind], np.ones((n + 1, 1))], axis=1)
            y_=y[rand_ind]
            v=None
            try:
                v=np.linalg.inv(A) @ y_
            except np.linalg.LinAlgError as err:
                pass
            if v is not None:
                eval=np.concatenate([X, np.ones((m, 1))], axis=1) @ v - y
                count=np.sum(np.abs(eval) <= self.margin)
                if self.best_params is None:
                    self.best_params=v
                    best_count=count
                else:
                    if count > best_count:
                        self.best_params=v
                        best_count=count
    
    def predict(self, X):
        m,n=X.shape
        pred=np.concatenate([X, np.ones((m, 1))], axis=1) @ self.best_params
        return pred

    def score(self, X, y, metric="mse"):
        loss=None
        if metric == 'mae':
            loss=MeanAbsoluteError()
        if metric == 'mse':
            loss=MeanSquareError()
        if metric == 'rmse':
            loss=RootMeanSquareError()
        s_z=Some(arr=self.predict(X))
        s_L=loss.get_loss(s_z, y)
        return s_L.arr[0,0]