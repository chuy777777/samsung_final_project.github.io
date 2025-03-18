import numpy as np
from framework_ml.puromachine import *

"""
Funciones de perdida
    - Clasificacion
        - Binary Cross Entropy (BCE)
        - Categorical Cross Entropy (CCE)
    - Regresion
        - Mean Absolute Error (MAE)
        - Mean Square Error (MSE)
        - Root Mean Square Error (RMSE)
"""

class LossFunctions():
    @staticmethod
    def get_loss_function(loss_function):
        loss_function_name=loss_function['name']
        loss_function_parameters=loss_function['parameters']
        if loss_function_name == 'bce':
            return BinaryCrossEntropy(**loss_function_parameters)
        elif loss_function_name == 'cce':
            return CategoricalCrossEntropy(**loss_function_parameters)
        elif loss_function_name == 'mae':
            return MeanAbsoluteError(**loss_function_parameters)
        elif loss_function_name == 'mse':
            return MeanSquareError(**loss_function_parameters)
        elif loss_function_name == 'rmse':
            return RootMeanSquareError(**loss_function_parameters)

""" 
Funciones de perdida para clasificacion 
"""
class BinaryCrossEntropy():
    def __init__(self, weights):
        # Diccionario con pesos para cada clase
        self.weights=weights
        self.name="bce"

    def get_loss(self, s_a, y):
        m=y.size
        b=np.zeros(m)        
        for key in self.weights.keys():
            b[y.flatten() == int(key)]=self.weights[key]
        s_B=Some(arr=np.diag(b))
        s_o=Some(arr=np.ones((m,1)))
        s_y=Some(arr=y)
        s_p=Some.matmul(Some.transpose(s_y), s_B)
        s_q=Some.matmul(s_p, Some.function(s_a, LnFunction))
        s_r=Some.sub(s_o, s_y)
        s_s=Some.matmul(Some.transpose(s_r), s_B)
        s_t=Some.matmul(s_s, Some.function(Some.sub(s_o, s_a), LnFunction))
        s_u=Some.add(s_q, s_t)
        s_L=Some.scalar_mul(- 1 / m, s_u)
        return s_L

class CategoricalCrossEntropy():
    def __init__(self, weights):
        # Diccionario con pesos para cada clase
        self.weights=weights
        self.name="cce"

    def get_loss(self, s_A, y):
        # NOTA: SE PUEDE HACER MEJOR (NO ES MUY EFICIENTE)
        m=y.size
        k=len(self.weights.keys())
        Y=np.zeros((k, m), dtype=int)
        Y[y.flatten(), np.arange(m)] = 1 
        s_Y=Some(arr=Y)
        p=np.zeros(m)        
        for key in self.weights.keys():
            p[y.flatten() == int(key)]=self.weights[key]
        s_P=Some(arr=np.diag(p))
        s_C=Some.matmul(s_A, s_Y)
        s_D=Some.function(s_C, LnFunction)
        s_E=Some.matmul(s_P, s_D)
        s_G=Some.function(s_E, TraceFunction)
        s_L=Some.scalar_mul(- 1 / m, s_G)
        return s_L

""" 
Funciones de perdida para regresion 
"""
class MeanAbsoluteError():
    def __init__(self):
        self.name="mae"

    def get_loss(self, s_z, y):
        m=y.size
        s_y=Some(arr=y)
        s_e=Some.sub(s_z, s_y)
        s_p=Some.function(s_e, AbsoluteValueFunction)
        s_q=Some.function(s_p, SumFunction)
        s_L=Some.scalar_mul(1 / m, s_q)
        return s_L
        
class MeanSquareError():
    def __init__(self):
        self.name="mse"

    def get_loss(self, s_z, y):
        m=y.size
        s_y=Some(arr=y)
        s_e=Some.sub(s_z, s_y)
        s_r=Some.matmul(Some.transpose(s_e), s_e)
        s_L=Some.scalar_mul(1 / (2 * m), s_r)
        return s_L

class RootMeanSquareError():
    def __init__(self):
        self.mse=MeanSquareError()
        self.name="rmse"

    def get_loss(self, s_z, y):
        s_q=self.mse.get_loss(s_z, y)
        s_L=Some.function(s_q, SquareRootFunction)
        return s_L