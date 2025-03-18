import numpy as np

# Toma un vector columna y lo devuelve a su forma original de dimensiones "shape"
def inv_vec(x, shape):
    if len(shape) == 0:   
        return x.squeeze()
    elif len(shape) == 1:   
        w=shape   
        return x.flatten()
    elif len(shape) == 2:   
        h,w=shape   
        return np.reshape(x, shape, order='F')
    elif len(shape) == 3:
        d,h,w=shape
        return np.transpose(np.reshape(x, (d,w,h), order='C'), axes=(0,2,1))
    elif len(shape) == 4:
        s,d,h,w=shape
        return np.transpose(np.reshape(x, (s,d,w,h), order='C'), axes=(0,1,3,2))

# Devuelve un vector columna de dimensiones (p,1)
def vec(X):
    if len(X.shape) == 0:   
        return np.expand_dims(X, axis=(0,1))
    elif len(X.shape) == 1:   
        return X[:,None]
    elif len(X.shape) == 2:      
        return X.flatten(order='F')[:,None]
    elif len(X.shape) == 3:
        return np.transpose(X, axes=(0,2,1)).flatten(order='C')[:,None]
    elif len(X.shape) == 4:
        return np.transpose(X, axes=(0,1,3,2)).flatten(order='C')[:,None]