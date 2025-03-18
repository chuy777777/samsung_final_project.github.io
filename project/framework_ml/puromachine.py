import numpy as np
from framework_ml.vectorization import vec, inv_vec

# Clase para la construccion del grafo computacional
class Some():
    def __init__(self, arr, calculate_grad=False, childs=[], operation='', dict_aux={}):
        # Para almacenar una matriz de Numpy de dimensiones (m,n)
        self.arr=arr
        # Para especificar si se desea calcular el gradiente para esta variable
        self.calculate_grad=calculate_grad
        # Para guardar otros objetos de tipo 'Some' de los cuales depende esta variable
        self.childs=childs
        # Para especificar que operacion se ha realizado para calcular esta variable
        self.operation=operation
        # Para guardar informacion de operaciones que lo necesiten
        self.dict_aux=dict_aux
        # Para guardar el gradiente de la variable en su forma vectorizada en forma de fila
        self.grad=None

    # Funcion auxiliar recursiva para el backpropagation
    def backward_recursive(self, father, dLdfather):
        if len(father.childs) > 0:
            for i in range(len(father.childs)):
                self.backward_recursive(father.childs[i], Some.derivative(dLdfather=dLdfather, father=father, child_index=i))
        else:
            if father.calculate_grad:
                father.grad=dLdfather if father.grad is None else father.grad + dLdfather

    # Funcion para iniciar el backpropagation
    def backward(self):
        m,n=self.arr.shape
        for i in range(len(self.childs)):
            self.backward_recursive(self.childs[i], Some.derivative(dLdfather=np.array([[1]]), father=self, child_index=i))

    """
    Devuelve dLdchild (derivada de la funcion de perdida con respecto a un nodo hijo de un nodo padre)
    
    Por lo general se tiene lo siguiente:
        dLdchild = dLdfather @ dfatherdchild
        (1,q)     (1,p)  (p,q)
    
        Observaciones: 
            - Ya se tiene 'dLdfather'
            - No se calcula explicitamente el jacobiano para 'dfatherdchild'
            - El objetivo es determinar 'dLdchild' sin tener que definir explicitamente a 'dfatherdchild' 
                En la experimentacion si se realiza la mutiplicacion pero solo una vez para sacar conclusiones 
                y patrones para calcular 'dLdchild' de una manera mas facil y rapida

    Se tienen operaciones:
        - Basicas
        - De funciones

    Por el momento, todo esto generaliza a cualquier matriz de dimensiones (h,w), por lo que se tiene que trabajar 
    con este tipo de matrices.

    Trabajo futuro: 
        Extender a matrices de dimensiones:
            - (d,h,w)
            - (s,d,h,w)
    """
    @staticmethod
    def derivative(dLdfather, father, child_index):
        operation=father.operation
        dLdchild=None
        if operation == 'matmul':
            # A (m,n), B (n,p)
            A,B=father.childs[0].arr,father.childs[1].arr
            if child_index == 0:
                dLdchild=(B @ np.reshape(dLdfather, (B.shape[1], A.shape[0]), order='C')).flatten(order='C')[None,:]
            else:
                dLdchild=(A.T @ np.reshape(dLdfather, (A.shape[0], B.shape[1]), order='F')).flatten(order='F')[None,:]
        if operation == 'mul':
            # A (m,n), B (m,n)
            A,B=father.childs[0].arr,father.childs[1].arr
            a,b=vec(A),vec(B)
            if child_index == 0:
                dLdchild=dLdfather * b.T
            else:
                dLdchild=dLdfather * a.T
        elif operation == 'add':
            # A (m,n), B (m,n)
            A,B=father.childs[0].arr,father.childs[1].arr
            if child_index == 0:
                dLdchild=dLdfather
            else:
                dLdchild=dLdfather
        elif operation == 'sub':
            # A (m,n), B (m,n)
            A,B=father.childs[0].arr,father.childs[1].arr
            if child_index == 0:
                dLdchild=dLdfather
            else:
                dLdchild=-dLdfather
        elif operation == 'scalar_mul':
            # A (m,n), scalar (valor numerico)
            A=father.childs[0].arr
            scalar=father.dict_aux['scalar']
            dLdchild=scalar * dLdfather
        elif operation == 'transpose':
            # A (m,n)
            A=father.childs[0].arr
            dLdchild=np.reshape(dLdfather, A.shape, order='C').flatten(order='F')[None,:]
        elif operation == 'function':
            # A (m,n)
            A=father.childs[0].arr
            F=father.arr
            function_class=father.dict_aux['function_class']
            dLdchild=function_class.derivative(dLdfather, A, F, **father.dict_aux['kwargs'])
        return dLdchild

    """
    Operaciones aplicables
    """
    @staticmethod
    # Multiplicacion matricial
    def matmul(s_A, s_B):
        A,B=s_A.arr,s_B.arr
        F=A @ B
        childs=[s_A, s_B]
        return Some(arr=F, childs=childs, operation='matmul')

    @staticmethod
    # Multiplicacion elemento a elemento (hadamard product)
    def mul(s_A, s_B):
        A,B=s_A.arr,s_B.arr
        F=A * B
        childs=[s_A, s_B]
        return Some(arr=F, childs=childs, operation='mul')

    @staticmethod
    # Adicion (suma)
    def add(s_A, s_B):
        A,B=s_A.arr,s_B.arr
        F=A + B
        childs=[s_A, s_B]
        return Some(arr=F, childs=childs, operation='add')

    @staticmethod
    # Substraccion (resta)
    def sub(s_A, s_B):
        A,B=s_A.arr,s_B.arr
        F=A - B
        childs=[s_A, s_B]
        return Some(arr=F, childs=childs, operation='sub')

    @staticmethod
    # Multiplicacion por un escalar
    def scalar_mul(scalar, s_A):
        A=s_A.arr
        F=scalar * A
        childs=[s_A]
        return Some(arr=F, childs=childs, operation='scalar_mul', dict_aux={'scalar': scalar})

    @staticmethod
    # Transpuesta 
    def transpose(s_A):
        A=s_A.arr
        F=A.T
        childs=[s_A]
        return Some(arr=F, childs=childs, operation='transpose')
        
    @staticmethod
    # Cualquier funcion 
    def function(s_A, function_class, **kwargs):
        A=s_A.arr
        F=function_class.evaluate(A, **kwargs)
        childs=[s_A]
        return Some(arr=F, childs=childs, operation='function', dict_aux={'function_class': function_class, 'kwargs': kwargs})

# Funciones aplicables en la construccion del grafo computacional
"""
Cada clase debe tener dos metodos:
    - evaluate
        Evalua la entrada con la funcion
    - derivative
        Esta pensado para el grafo computacional y devuelve dLdchild
"""
# Funcion traza
class TraceFunction():
    @staticmethod
    # A (m,m)
    def evaluate(A):
        F=np.array([[np.trace(A)]])
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        m,_=A.shape
        dLdchild=dLdfather @ vec(np.eye(m)).T
        return dLdchild

# Funcion potencia
class PowerFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A, b):
        F=A ** float(b)
        return F

    @staticmethod
    def derivative(dLdfather, A, F, b):
        a=vec(A)
        dLdchild=dLdfather * (float(b) * (a ** float(b - 1))).T
        return dLdchild

# Funcion softmax
class SoftmaxFunction():
    @staticmethod
    # A (m,k)
    def evaluate(A):
        F=np.exp(A) * (1 / np.sum(np.exp(A), axis=1)[:,None])
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        m,n=A.shape
        f=vec(F)
        B=f @ f.T
        C=np.kron(np.ones((n,n)), np.eye(m))
        D=B * C
        E=np.diag(f.flatten())
        G=E - D
        dfatherdchild=G
        dLdchild=dLdfather @ dfatherdchild
        return dLdchild

# Funcion sigmoide
class SigmoidFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A):
        eps=1e-4
        F=np.clip(1 / (1 + np.exp(-A)), 0 + eps, 1 - eps)
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        f=vec(F)
        dLdchild=dLdfather * (f * (1 - f)).T
        return dLdchild

# Funcion logaritmo natural
class LnFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A):
        F=np.log(A)
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        a=vec(A)
        dLdchild=dLdfather * (1 / a).T
        return dLdchild

# Funcion valor absoluto
class AbsoluteValueFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A):
        F=np.abs(A)
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        r=vec(A)
        r[r < 0]=-1
        r[r == 0]=0
        r[r > 0]=1
        dLdchild=dLdfather * r.T
        return dLdchild

# Funcion suma de todos los elementos 
class SumFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A):
        F=np.array([[np.sum(A)]])
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        dLdchild=dLdfather @ np.ones((1,A.size))
        return dLdchild

# Funcion raiz cuadrada
class SquareRootFunction():
    @staticmethod
    # A (m,n)
    def evaluate(A):
        F=np.sqrt(A)
        return F

    @staticmethod
    def derivative(dLdfather, A, F):
        a=vec(A)
        r=1 / (2 * np.sqrt(a))
        dLdchild=dLdfather * r.T
        return dLdchild