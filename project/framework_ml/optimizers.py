import numpy as np

"""
Optimizadores
    - Gradient Descent
        - Dependiendo el tamaño de datos con el que se trabaja a la vez:
            - Batch Gradient Descent (todo el conjunto de datos)
            - Mini-batch Gradient Descent (un subconjunto del conjunto de datos)
            - Stochastic Gradient Descent (un solo ejemplo a la vez)
        - Variantes:
            - With Momentum
            - Nesterov Accelerated Gradient
    - AdaGrad
    - AdaDelta
    - RMSProp
    - Adam
"""

class Optimizers():
    @staticmethod
    def get_optimizer(optimizer):
        optimizer_name=optimizer['name']
        optimizer_parameters=optimizer['parameters']
        if optimizer_name == 'gd':
            return GradientDescent(**optimizer_parameters)
        elif optimizer_name == 'adagrad':
            return AdaGrad(**optimizer_parameters)
        elif optimizer_name == 'adadelta':
            return AdaDelta(**optimizer_parameters)
        elif optimizer_name == 'rmsprop':
            return RMSProp(**optimizer_parameters)
        elif optimizer_name == 'adam':
            return Adam(**optimizer_parameters)

class GradientDescent():
    def __init__(self, learning_rate=0.1, momentum=0.90, nesterov=True):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.nesterov=nesterov
        self.name="gd"
        self.init()

    def init(self):
        self.it=0
        self.u=None

    def prev_operate(self, params):
        if self.momentum is None:
            return params
        else:
            if self.it == 0:
                self.u=np.zeros(params.size)
            if self.nesterov:
                v=params + self.momentum * self.u
                return v
            else:
                return params
    
    def operate(self, params, grad):
        new_params=np.zeros(params.size)
        if self.momentum is None:
            new_params=params + (- self.learning_rate * grad)
        else:
            if self.nesterov:
                grad_forward=grad
                self.u=(self.momentum * self.u) + (- self.learning_rate * grad_forward)
                new_params=params + self.u
            else:
                self.u=(self.momentum * self.u) + (- self.learning_rate * grad)
                new_params=params + self.u
        self.it+=1
        return new_params

class AdaGrad():
    def __init__(self, learning_rate=0.1, eps=1e-4):
        self.learning_rate=learning_rate
        self.eps=eps
        self.name="adagrad"
        self.init()

    def init(self):
        self.it=0
        self.v=None

    def prev_operate(self, params):
        return params
    
    def operate(self, params, grad):
        new_params=np.zeros(params.size)
        if self.it == 0:
            self.v=0
        self.v=self.v + grad ** 2
        new_params=params + ((1 / (np.sqrt(self.v) + self.eps)) * (- self.learning_rate * grad))
        self.it+=1
        return new_params

class AdaDelta():
    def __init__(self, momentum=0.90, initial_u=1e-6, eps=1e-4):
        self.momentum=momentum
        self.initial_u=initial_u
        self.eps=eps
        self.name="adadelta"
        self.init()

    def init(self):
        self.it=0
        self.u=None
        self.v=None
        self.prev_params=None

    def prev_operate(self, params):
        return params
    
    def operate(self, params, grad):
        new_params=np.zeros(params.size)
        if self.it == 0:
            self.v=0
            self.u=self.initial_u
        self.v=(1 - self.momentum) * grad ** 2 + self.momentum * self.v
        if self.it > 0:
            delta_params=params - self.prev_params
            self.u=(1 - self.momentum) * delta_params ** 2 + self.momentum * self.u
        new_params=params + ((1 / (np.sqrt(self.v) + self.eps)) * (- np.sqrt(self.u) * grad))
        self.prev_params=params
        self.it+=1
        return new_params

class RMSProp():
    def __init__(self, learning_rate=0.1, momentum=0.90, eps=1e-4):
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.eps=eps
        self.name="rmsprop"
        self.init()

    def init(self):
        self.it=0
        self.v=None

    def prev_operate(self, params):
        return params
    
    def operate(self, params, grad):
        new_params=np.zeros(params.size)
        if self.it == 0:
            self.v=0
        self.v=(1 - self.momentum) * grad ** 2 + self.momentum * self.v
        new_params=params + ((1 / (np.sqrt(self.v) + self.eps)) * (- self.learning_rate * grad))
        self.it+=1
        return new_params

class Adam():
    def __init__(self, learning_rate=0.1, beta_1=0.90, beta_2=0.999, eps=1e-4):
        self.learning_rate=learning_rate
        self.beta_1=beta_1
        self.beta_2=beta_2
        self.eps=eps
        self.name="adam"
        self.init()

    def init(self):
        self.it=0
        self.v=None
        self.u=None

    def prev_operate(self, params):
        return params
    
    def operate(self, params, grad):
        new_params=np.zeros(params.size)
        if self.it == 0:
            self.v=0
            self.u=0
        self.v=(1 - self.beta_2) * grad ** 2 + self.beta_2 * self.v
        self.u=(1 - self.beta_1) * grad + self.beta_1 * self.u
        self.v_=self.v / (1 - self.beta_2 ** (self.it + 1))
        self.u_=self.u / (1 - self.beta_1 ** (self.it + 1))
        new_params=params + ((1 / (np.sqrt(self.v_) + self.eps)) * (- self.learning_rate * self.u_))
        self.it+=1
        return new_params