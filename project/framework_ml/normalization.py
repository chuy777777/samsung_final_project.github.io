import numpy as np

# Normalizacion estandar 
# z = (x - mean) / std
class StandardNormalization():
    def __init__(self, mean=None, std=None):
        self.mean=mean
        self.std=std

    def fit(self, X):
        self.mean=X.mean(axis=0)
        self.std=X.std(axis=0, ddof=0)

    def normalize(self, X):
        m,n=X.shape
        Z=np.zeros((m,n))
        for col in range(n):
            if self.std[col] != 0:
                Z[:,col]=(X[:,col] - self.mean[col]) / self.std[col]
        return Z

    def normalize_inv(self, Z):
        m,n=Z.shape
        X=np.zeros((m,n))
        for col in range(n):
            X[:,col]= self.mean[col] + self.std[col] * Z[:,col]
        return X