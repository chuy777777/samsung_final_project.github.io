import numpy as np
import os
import pickle
from framework_ml.utils import FitProcess, ConfusionMatrix
from framework_ml.vectorization import vec, inv_vec
from framework_ml.optimizers import Optimizers
from framework_ml.loss_functions import LossFunctions
from framework_ml.puromachine import *

class Sequential():
    def __init__(
            self, 
            k=3,
            layers=[],
            reset_parameters=True,
            optimizer={"name": "", "parameters": {}}, 
            loss_function={"name": "", "parameters": {}},
        ):
        # Se establecen los parametros de clase unicos
        # Numero de clases
        self.k=k
        # Capas
        self.layers=layers
        # Para especificar si se desean volver a iniciar los pesos del modelo cada vez que se ejecuta la funcion 'fit'
        self.reset_parameters=reset_parameters
        # Se establecen los parametros de clase (hiperparametros)
        self.set_class_parameters(optimizer, loss_function)
        self.name='sequential'
        
    def set_class_parameters(self, optimizer, loss_function):
        # Optimizador a utilizar
        self.optimizer=Optimizers.get_optimizer(optimizer)
        # Funcion de perdida a utilizar
        self.loss_function=LossFunctions.get_loss_function(loss_function)

    def params_initialized(self):
        initialized=[layer.params_initialized() for layer in self.layers]
        return np.sum(initialized) == len(self.layers)

    def reset_grad(self):
        for layer in self.layers:
            layer.reset_grad()
        
    def get_grad(self):
        grad=None
        for layer in self.layers:
            grad=layer.get_grad() if grad is None else np.concatenate([grad, layer.get_grad()])
        return grad

    def get_params(self):
        params=None
        for layer in self.layers:
            params=layer.get_params() if params is None else np.concatenate([params, layer.get_params()])
        return params

    def set_params(self, new_params):
        start=0
        stop=0
        for layer in self.layers:
            stop+=layer.n_input_features * layer.n_neurons + layer.n_neurons
            layer.set_params(new_params[start:stop])
            start=stop

    def fit(self, X, y, params_init, epochs=100, tol=1e-4, num_mini_batches=1):
        history,stopped=FitProcess.supervised_learning_fit(X, y, self, params_init, epochs, tol, num_mini_batches)
        return history,stopped

    def build_computational_graph(self, X, y):
        # Prediccion del modelo
        s_A=self.graph_prediction(X)
        # Funcion de perdida (cuantifica el error entre lo predicho y lo real)
        s_L=self.loss_function.get_loss(s_A, y)
        return s_L
    
    def graph_prediction(self, X):
        # Prediccion del modelo para el conjunto de datos proporcionado (en forma de grafo)
        s_A=Some(arr=X)
        for layer in self.layers:
            s_A=layer.evaluate(s_A)
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

    def save(self, path, file_name):
        full_path=os.path.join(path, *[f"{file_name}.pkl"])
        if not os.path.exists(path):
            os.makedirs(path)
        data={
            "k": self.k,
            "layers": self.layers,
            "reset_parameters": self.reset_parameters,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function
        }
        with open(full_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path, file_name):
        full_path=os.path.join(path, *[f"{file_name}.pkl"])
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                data=pickle.load(f)
                model=Sequential(
                    k=data["k"], 
                    layers=data["layers"], 
                    reset_parameters=data["reset_parameters"],
                )
                model.optimizer=data["optimizer"]
                model.loss_function=data["loss_function"]
                return model
        return None

class FCLayer():
    def __init__(self, n_input_features, n_neurons, activation_function):
        self.n_input_features=n_input_features
        self.n_neurons=n_neurons
        self.activation_function=activation_function
        # Pesos de la capa
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

    def evaluate(self, s_A_prev):
        m,_=s_A_prev.arr.shape
        s_o=Some(arr=np.ones((m,1)))
        s_U=Some.matmul(s_A_prev, self.s_W)
        s_B=Some.matmul(s_o, Some.transpose(self.s_b))
        s_Z=Some.add(s_U, s_B)
        s_A_next=Some.function(s_Z, self.activation_function)
        return s_A_next