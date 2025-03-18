import numpy as np
from framework_ml.utils import FitProcess, ConfusionMatrix
from framework_ml.vectorization import vec, inv_vec
from framework_ml.optimizers import Optimizers
from framework_ml.loss_functions import LossFunctions
from framework_ml.puromachine import *
from framework_ml.regularization import Regularization

class SoftmaxRegression():
    def __init__(
            self, 
            k=3,
            reset_parameters=True,
            regularization="l2", 
            regularization_term=0.01, 
            optimizer={"name": "", "parameters": {}}, 
            loss_function={"name": "", "parameters": {}},
        ):
        # Se establecen los parametros de clase unicos
        # Numero de clases
        self.k=k
        # Para especificar si se desean volver a iniciar los pesos del modelo cada vez que se ejecuta la funcion 'fit'
        self.reset_parameters=reset_parameters
        # Se establecen los parametros de clase (hiperparametros)
        self.set_class_parameters(regularization, regularization_term, optimizer, loss_function)
        self.name='softmax_regression'
        
    def set_class_parameters(self, regularization, regularization_term, optimizer, loss_function):
        # Tipo de regularizacion ["l1", "l2"]
        self.regularization=regularization
        # Parametro de regularizacion 
        self.regularization_term=regularization_term
        # Optimizador a utilizar
        self.optimizer=Optimizers.get_optimizer(optimizer)
        # Funcion de perdida a utilizar
        self.loss_function=LossFunctions.get_loss_function(loss_function)
        # Pesos de los modelos lineales (un modelo por cada clase)
        self.s_W=None
        self.s_b=None

    def params_initialized(self):
        return (self.s_W is not None and self.s_b is not None)

    def reset_grad(self):
        self.s_W.grad=None
        self.s_b.grad=None
        
    def get_grad(self):
        grad=np.concatenate([self.s_W.grad, self.s_b.grad], axis=1).flatten()
        return grad

    def get_params(self):
        params=np.concatenate([vec(self.s_W.arr), vec(self.s_b.arr)], axis=0).flatten()
        return params

    def set_params(self, new_params):
        self.s_W.arr=inv_vec(new_params[0:self.s_W.arr.size][:,None], self.s_W.arr.shape)
        self.s_b.arr=inv_vec(new_params[self.s_W.arr.size:][:,None], self.s_b.arr.shape)

    def fit(self, X, y, params_init, epochs=100, tol=1e-4, num_mini_batches=1):
        history,stopped=FitProcess.supervised_learning_fit(X, y, self, params_init, epochs, tol, num_mini_batches)
        return history,stopped

    def build_computational_graph(self, X, y):
        # Prediccion del modelo
        s_A=self.graph_prediction(X)
        # Funcion de perdida (cuantifica el error entre lo predicho y lo real)
        s_L=self.loss_function.get_loss(s_A, y)
        # Se agrega la regularizacion en caso de que se haya especificado
        if self.regularization is not None:
            s_L=Regularization.add_l_regularization(self.regularization, self.regularization_term, s_L, self.s_W)
        return s_L
    
    def graph_prediction(self, X):
        # Prediccion del modelo para el conjunto de datos proporcionado (en forma de grafo)
        m,n=X.shape
        s_X=Some(arr=X)
        s_o=Some(arr=np.ones((m,1)))
        s_U=Some.matmul(s_X, self.s_W)
        s_B=Some.matmul(s_o, Some.transpose(self.s_b))
        s_Z=Some.add(s_U, s_B)
        s_A=Some.function(s_Z, SoftmaxFunction)
        return s_A
        
    def predict(self, X):
        # Predicciones (solo las predicciones de clase)
        s_A=self.graph_prediction(X)
        pred=np.argmax(s_A.arr, axis=1)[:,None]
        return pred

    def predict_proba(self, X):
        # Predicciones (solo las probabilidades)
        s_A=self.graph_prediction(X)
        return s_A.arr

    def score(self, X, y, metric="f1_score"):
        pred=self.predict(X)
        confusion_matrix=ConfusionMatrix(pred, y, k=self.k)
        return confusion_matrix.get_score(metric)