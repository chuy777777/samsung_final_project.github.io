import numpy as np
from framework_ml.puromachine import Some
from framework_ml.loss_functions import MeanAbsoluteError, MeanSquareError, RootMeanSquareError

class PolynomialRegression():
    def __init__(self, p):
        self.p=p
        self.params=None

    def generate_poly_matrix(self, X):
        m,n=X.shape
        if n == 1:
            X_=np.zeros((m,self.p))
            for i in range(self.p):
                X_[:,i]=X[:,0] ** (i + 1)
            return X_
    
    def fit(self, X, y):
        m,n=X.shape
        X_=self.generate_poly_matrix(X)
        A=np.concatenate([X_, np.ones((m, 1))], axis=1)
        try:
            v=np.linalg.inv(A.T @ A) @ A.T @ y
            self.params=v
        except np.linalg.LinAlgError as err:
            self.params=None
    
    def predict(self, X):
        m,n=X.shape
        X_=self.generate_poly_matrix(X)
        pred=np.concatenate([X_, np.ones((m, 1))], axis=1) @ self.params
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