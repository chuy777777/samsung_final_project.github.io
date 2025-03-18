import pickle
import os 

"""
Clase para guardar los parametros de optimizacion.

Estos parametros son utilizados en el proceso de optimizacion para encontrar los parametros mas optimos de la camara.

Los parametros corresponden a los utilizados en el optimizador 'Adam'.
"""
class OptimizationParameters():
    def __init__(self, num_it, lr, beta_1, beta_2):
        self.num_it=num_it
        self.lr=lr
        self.beta_1=beta_1
        self.beta_2=beta_2

    # Para guardar los parametros de optimizacion en la computadora
    def save(self, full_path):
        with open(os.path.join(full_path, *["num_it.pickle"]), 'wb') as file:
            pickle.dump(self.num_it, file)
        with open(os.path.join(full_path, *["lr.pickle"]), 'wb') as file:
            pickle.dump(self.lr, file)
        with open(os.path.join(full_path, *["beta_1.pickle"]), 'wb') as file:
            pickle.dump(self.beta_1, file)
        with open(os.path.join(full_path, *["beta_2.pickle"]), 'wb') as file:
            pickle.dump(self.beta_2, file)

    # Para cargar los parametros de optimizacion 
    @staticmethod
    def load(full_path):
        num_it=None 
        lr=None
        beta_1=None
        beta_2=None
        with open(os.path.join(full_path, *["num_it.pickle"]), 'rb') as file:
            num_it=pickle.load(file)
        with open(os.path.join(full_path, *["lr.pickle"]), 'rb') as file:
            lr=pickle.load(file)
        with open(os.path.join(full_path, *["beta_1.pickle"]), 'rb') as file:
            beta_1=pickle.load(file)
        with open(os.path.join(full_path, *["beta_2.pickle"]), 'rb') as file:
            beta_2=pickle.load(file)
        return OptimizationParameters(num_it=num_it, lr=lr, beta_1=beta_1, beta_2=beta_2)

    # Para imprimir los parametros de optimizacion
    def __str__(self):
        return "Number of iterations: {}\nLearning rate: {}\nBeta 1: {}\nBeta 2: {}".format(self.num_it, self.lr, self.beta_1, self.beta_2)