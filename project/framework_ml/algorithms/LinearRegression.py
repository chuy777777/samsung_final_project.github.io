import numpy as np
from framework_ml.utils import FitProcess
from framework_ml.vectorization import vec, inv_vec
from framework_ml.optimizers import Optimizers
from framework_ml.loss_functions import LossFunctions
from framework_ml.puromachine import *
from framework_ml.regularization import Regularization

class LinearRegression():
    def __init__(
            self, 
            reset_parameters=True,
            regularization="l2", 
            regularization_term=0.01, 
            optimizer={"name": "", "parameters": {}}, 
            loss_function={"name": "", "parameters": {}}, 
        ):
        # Se establecen los parametros de clase unicos
        # Para especificar si se desean volver a iniciar los pesos del modelo cada vez que se ejecuta la funcion 'fit'
        self.reset_parameters=reset_parameters
        # Se establecen los parametros de clase (hiperparametros)
        self.set_class_parameters(regularization, regularization_term, optimizer, loss_function)
        self.name='linear_regression'
        
    def set_class_parameters(self, regularization, regularization_term, optimizer, loss_function):
        # Tipo de regularizacion ["l1", "l2"]
        self.regularization=regularization
        # Parametro de regularizacion 
        self.regularization_term=regularization_term
        # Optimizador a utilizar
        self.optimizer=Optimizers.get_optimizer(optimizer)
        # Funcion de perdida a utilizar
        self.loss_function=LossFunctions.get_loss_function(loss_function)
        # Pesos del modelo (modelo lineal)
        self.s_w=None
        self.s_b=None

    def params_initialized(self):
        return (self.s_w is not None and self.s_b is not None)

    def reset_grad(self):
        self.s_w.grad=None
        self.s_b.grad=None
    
    def get_grad(self):
        grad=np.concatenate([self.s_w.grad, self.s_b.grad], axis=1).flatten()
        return grad

    def get_params(self):
        params=np.concatenate([vec(self.s_w.arr), vec(self.s_b.arr)], axis=0).flatten()
        return params

    def set_params(self, new_params):
        self.s_w.arr=inv_vec(new_params[0:self.s_w.arr.size][:,None], self.s_w.arr.shape)
        self.s_b.arr=inv_vec(new_params[self.s_w.arr.size:][:,None], self.s_b.arr.shape)

    def fit(self, X, y, params_init, epochs=100, tol=1e-4, num_mini_batches=1):
        history,stopped=FitProcess.supervised_learning_fit(X, y, self, params_init, epochs, tol, num_mini_batches)
        return history,stopped

    def build_computational_graph(self, X, y):
        # Prediccion del modelo
        s_z=self.graph_prediction(X)
        # Funcion de perdida (cuantifica el error entre lo predicho y lo real)
        s_L=self.loss_function.get_loss(s_z, y)
        # Se agrega la regularizacion en caso de que se haya especificado
        if self.regularization is not None:
            s_L=Regularization.add_l_regularization(self.regularization, self.regularization_term, s_L, self.s_w)
        return s_L
    
    def graph_prediction(self, X):
        # Prediccion del modelo para el conjunto de datos proporcionado (en forma de grafo)
        m,n=X.shape
        s_X=Some(arr=X)
        s_o=Some(arr=np.ones((m,1)))
        s_r=Some.matmul(s_X, self.s_w)
        s_v=Some.matmul(s_o, self.s_b)
        s_z=Some.add(s_r, s_v)
        return s_z
        
    def predict(self, X):
        # Prediccion del modelo para el conjunto de datos proporcionado (solo las predicciones)
        s_z=self.graph_prediction(X)
        return s_z.arr

    def score(self, X, y):
        # Costo de la funcion de perdida para los datos proporcionados
        s_L=self.build_computational_graph(X, y)
        loss=s_L.arr[0,0]
        return loss