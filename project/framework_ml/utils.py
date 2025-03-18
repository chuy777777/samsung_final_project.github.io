import numpy as np
from sympy.core.symbol import Symbol
import matplotlib.pyplot as plt

# Variables simbolicas
def symbolic_variable(shape, letter='x'):
    X=np.zeros(shape, dtype='object')
    m,n=shape
    count=1
    for j in range(n):
        for i in range(m):
            X[i,j]=Symbol(f'{letter}_{count}')
            count+=1
    return X
    
# Dividir conjunto de datos en conjunto de entrenamiento y conjunto de prueba
def split_train_test(X, y, train=0.8, random=False, seed=None, cv=None):
    np.random.seed(seed)
    X_=np.copy(X)
    y_=np.copy(y)
    m,n=X_.shape
    if random:
        indexes=np.random.choice(m, m, replace=False)
        X_=X_[indexes]
        y_=y_[indexes]
        
    if cv is None:
        train_size=int(m * train)
        X_train,y_train=X_[0:train_size],y_[0:train_size]
        X_test,y_test=X_[train_size:],y_[train_size:]
        
        return X_train, X_test, y_train, y_test
    else:
        classes,m_classes=np.unique(y, return_counts=True)
        k=len(classes)
        
        # Lista de matrices de instancias por clase
        X_classes=[X[y.flatten() == classes[i]] for i in range(k)]
        
        """
        En el conjunto de entrenamiento:
            - 'train' % de toda la clase 0
            - 'train' % de toda la clase 1
            - ...
            - 'train' % de toda la clase k - 1
        
        En el conjunto de prueba:
            - '1 - train' % de toda la clase 0
            - '1 - train' % de toda la clase 1
            - ...
            - '1 - train' % de toda la clase k - 1
        """
        X_train=np.concatenate([X_classes[i][0:int(m_classes[i] * train)] for i in range(k)], axis=0)
        X_test=np.concatenate([X_classes[i][int(m_classes[i] * train):] for i in range(k)], axis=0)
        y_train=np.concatenate([(np.ones((X_classes[i][0:int(m_classes[i] * train)].shape[0], 1)) * i).astype(int) for i in range(k)], axis=0)
        y_test=np.concatenate([(np.ones((X_classes[i][int(m_classes[i] * train):].shape[0], 1)) * i).astype(int) for i in range(k)], axis=0)
        
        classes,m_classes=np.unique(y_train, return_counts=True)
        # Lista de matrices de instancias por clase (conjunto de entrenamiento)
        X_classes=[X_train[y_train.flatten() == classes[i]] for i in range(k)]

        """
        En el conjunto de entrenamiento (todos los subconjuntos estan colocados secuencialmente): 
            - Subconjunto '1'
                - '1/cv * 100' % de toda la clase 0 (en el conjunto de entrenamiento)
                - '1/cv * 100' % de toda la clase 1 (en el conjunto de entrenamiento)
                - ...
                - '1/cv * 100' % de toda la clase k - 1 (en el conjunto de entrenamiento)
            - ...
            - Subconjunto 'cv' 
                - '1/cv * 100' % de toda la clase 0 (en el conjunto de entrenamiento)
                - '1/cv * 100' % de toda la clase 1 (en el conjunto de entrenamiento)
                - ...
                - '1/cv * 100' % de toda la clase k - 1 (en el conjunto de entrenamiento)
        """
        X_train_temp=None
        y_train_temp=None
        for j in range(cv):
            X_cv=np.concatenate([X_classes[i][int((m_classes[i] / cv) * j):int((m_classes[i] / cv) * (j + 1))] for i in range(k)], axis=0)
            X_train_temp=X_cv if X_train_temp is None else np.concatenate([X_train_temp, X_cv], axis=0)
            y_cv=np.concatenate([(np.ones((X_classes[i][int((m_classes[i] / cv) * j):int((m_classes[i] / cv) * (j + 1))].shape[0], 1)) * i).astype(int) for i in range(k)], axis=0)
            y_train_temp=y_cv if y_train_temp is None else np.concatenate([y_train_temp, y_cv], axis=0)

        return X_train_temp, X_test, y_train_temp, y_test
        
# Dividir conjunto de datos en varios subconjuntos del mismo tamaño (con opcion de que sea de forma aleatoria)
def split_mini_batches(X, y, num_mini_batches, random=False, seed=None):
    np.random.seed(seed)
    X_=np.copy(X)
    y_=np.copy(y)
    m,n=X_.shape
    if random:
        indexes=np.random.choice(m, m, replace=False)
        X_=X_[indexes]
        y_=y_[indexes]
    size=int(np.round(m / num_mini_batches))
    mini_batches=[]
    for i in range(num_mini_batches):
        mini_batches.append((X_[i * size:(i + 1) * size], y_[i * size:(i + 1) * size]))
    return mini_batches

# Clase auxiliar para la generacion de combinaciones de parametros de clase
class ClassParametersNode():
    def __init__(self, values, node, model_param, name=None, deep_param=None):
        self.values=values
        self.node=node
        self.model_param=model_param
        self.name=name
        self.deep_param=deep_param

    def __str__(self):
        return f'{self.model_param} - {self.name} - {self.deep_param}'
        
# Devuelve diferentes combinaciones de valores para cada parametro de clase
def get_class_parameters_combinations_1(class_parameters_dict_list):
    def recursive(node, l_comb, comb):
        if node is not None:
            for value in node.values:
                recursive(node.node, l_comb, comb + [(value, node)])
        else:
            l_comb.append(comb)

    temp=len(class_parameters_dict_list['optimizer']),len(class_parameters_dict_list['loss_function'])
    l_indexes=[]
    for i in range(temp[0]):
        for j in range(temp[1]):
            l_indexes.append([i,j])

    l_obj=[]
    for indexes in l_indexes:
        node=None
        params1=['optimizer', 'loss_function']
        params2=list(set(class_parameters_dict_list.keys()).difference(params1))
        for model_param in params2:
            values=class_parameters_dict_list[model_param]
            node=ClassParametersNode(values, node, model_param)
    
        empty=[]
        for i,model_param in enumerate(params1):
            obj=class_parameters_dict_list[model_param][indexes[i]]
            if obj['parameters']:
                for deep_param in obj['parameters'].keys():
                    values=obj['parameters'][deep_param]
                    node=ClassParametersNode(values, node, model_param, obj['name'], deep_param)
            else:
                empty.append((model_param, obj['name'], i))

        l_comb=[]
        recursive(node, l_comb, [])
    
        for comb in l_comb:
            obj={}
            for value, node in comb:
                if node.model_param not in params1:
                    obj[node.model_param]=value
                else:
                    if node.model_param not in obj:
                        obj[node.model_param]={"name": node.name, "parameters": {}}
                    obj[node.model_param]['parameters'][node.deep_param]=value
            for model_param, name, i in empty:
                obj[model_param]=class_parameters_dict_list[model_param][indexes[i]]
            l_obj.append(obj)
        
    return l_obj

# Devuelve diferentes combinaciones de valores para cada parametro de clase
def get_class_parameters_combinations_2(class_parameters_dict_list):
    def recursive(node, l_comb, comb):
        if node is not None:
            for value in node.values:
                recursive(node.node, l_comb, comb + [(value, node)])
        else:
            l_comb.append(comb)

    node=None
    for model_param in class_parameters_dict_list.keys():
        values=class_parameters_dict_list[model_param]
        node=ClassParametersNode(values, node, model_param)
        
    l_comb=[]
    recursive(node, l_comb, [])

    l_obj=[]
    for comb in l_comb:
        obj={}
        for value, node in comb:
            obj[node.model_param]=value
        l_obj.append(obj)
    
    return l_obj

"""
Clase para entrenar un modelo con diferentes combinaciones de parametros de clase (hiperparametros) utilizando 
'cross validation' para cada combinacion.

Al final se construye el mejor modelo con los mejores parametros de clase (hiperparametros).

La estrategia 'cross validation' es la siguiente:
    - Se divide el conjunto de datos en 'cv' subconjuntos del mismo tamaño
    - Se entrena el modelo con 'cv - 1' subconjuntos
    - Se prueba el modelo con el subconjunto restante
    - Se repite el proceso 'cv' veces con diferentes subconjuntos cada vez

Esta estrategia garantiza que el entrenamiento sea independiente de la particion de datos que se uso para
su entrenamiento, ya que se entrena con diferentes subconjuntos cada vez.
"""
class GridSearchCV():
    def __init__(self, model, class_parameters_dict_list):
        # Modelo que se utiliza
        self.model=model
        # Diccionario de parametros de clase en forma de lista para los valores
        self.class_parameters_dict_list=class_parameters_dict_list
        # Obtencion de todas las combinaciones de los parametros de clase (hiperparametros)
        self.class_parameters_combinations=get_class_parameters_combinations_1(self.class_parameters_dict_list)
        # Inicializa variables
        self.init()

    def init(self):
        # Historial de la funcion de perdida (en caso de que se haya especificado 're_train_best_model')
        self.history=None
        # Indica si el proceso de optimizacion se detuvo antes (en caso de que se haya especificado 're_train_best_model')
        self.stopped=None
        # Para guardar los mejores parametros de clase (hiperparametros)
        self.best_class_parameters=None
        # Para guardar el indice de la mejor combinacion
        self.best_index_comb=None
        # Para guardar los 'scores' de cada combinacion  
        self.scores_by_comb_dict={}

    def fit(self, X, y, params_init, score_metric=None, cv=5, re_train_best_model=True, epochs=100, tol=1e-4, num_mini_batches=1, highest_score=True):
        # Inicializacion de variables
        self.init()
        m,n=X.shape
        # Tamaño de cada subconjunto 
        size=int(m / cv)

        # Para cada combinacion se realiza 'cross validation'
        for index_comb,class_parameters_combination in enumerate(self.class_parameters_combinations):
            # Se establecen los parametros de clase (hiperparametros) al modelo 
            self.model.set_class_parameters(**class_parameters_combination)
            scores=[]
            # 'cross validation'
            for i in range(cv):
                # Se define el conjunto de entrenamiento y el conjunto de prueba
                X_train=X[size:] if i == 0 else np.concatenate([X[0:i * size], X[(i + 1) * size:]], axis=0)
                y_train=y[size:] if i == 0 else np.concatenate([y[0:i * size], y[(i + 1) * size:]], axis=0)
                X_test=X[i * size:(i + 1) * size]
                y_test=y[i * size:(i + 1) * size]
                # Se entrena el modelo con el conjunto de entrenamiento
                _,_=self.model.fit(X_train, y_train, params_init=params_init, epochs=epochs, tol=tol, num_mini_batches=num_mini_batches)
                # Se obtiene el 'score' con el conjunto de prueba
                if score_metric is not None:
                    score=self.model.score(X_test, y_test, score_metric)
                else:
                    score=self.model.score(X_test, y_test)
                # Se guarda el 'score'
                scores.append(score)
            # Se guardan los 'score' de cada iteracion en 'cross validation'
            self.scores_by_comb_dict[index_comb]=scores

        """
        En base a los resultados, se eligen los mejores parametros de clase (hiperparametros)
        de acuerdo a 'highest_score':
            - Si 'highest_score=True', se elige el que obtuvo un score promedio mas alto
            - Si 'highest_score=False', se elige el que obtuvo un score promedio mas bajo
        """
        for new_index_comb in self.scores_by_comb_dict.keys():
            if self.best_index_comb is None:
                self.best_index_comb=new_index_comb 
            else:
                best_scores=self.scores_by_comb_dict[self.best_index_comb]
                new_scores=self.scores_by_comb_dict[new_index_comb]
                best_mean_score=np.mean(best_scores)
                new_mean_score=np.mean(new_scores)
                if highest_score:
                    if new_mean_score > best_mean_score:
                        self.best_index_comb=new_index_comb
                else:
                    if new_mean_score < best_mean_score:
                        self.best_index_comb=new_index_comb

        # Se guardan los mejores parametros de clase (hiperparametros)
        self.best_class_parameters=self.class_parameters_combinations[self.best_index_comb]
        # Se establecen los mejores parametros de clase (hiperparametros) al modelo
        self.model.set_class_parameters(**self.best_class_parameters)
        # Si se especifica se vuelve a entrenar el modelo pero ahora con todo el conjunto de datos proporcionado
        if re_train_best_model:
            history,stopped=self.model.fit(X, y, params_init=params_init, epochs=epochs, tol=tol, num_mini_batches=1)
            self.history=history
            self.stopped=stopped   

class FitProcess():
    @staticmethod
    def supervised_learning_fit(X, y, model, params_init, epochs=100, tol=1e-4, num_mini_batches=1):
        # Inicializacion de parametros 
        if model.reset_parameters or not model.params_initialized():
            params_init(X, y, model)
        # Inicializacion del optimizador
        model.optimizer.init()
        # Creacion de mini-batches (se trabaja con cada mini-batch a la vez)
        # Esto es util cuando es muy costoso evaluar la funcion de perdida con todo el conjunto de datos
        mini_batches=split_mini_batches(X, y, num_mini_batches)
        # Para guardar el historial del costo de la funcion de perdida en cada epoca
        history=np.zeros(epochs)
        # Para especificar cuando se ha cumplido la tolerancia
        stopped=False

        # Iteracion a travez de cada epoca (una epoca equivale a trabajar con todo el conjunto de datos)
        for epoch in range(epochs):
            history_epoch=np.zeros(num_mini_batches)
            for it_mb,mini_batch in enumerate(mini_batches):
                # Actualizacion de parametros antes de la construccion del grafo computacional (opcional)
                params=model.get_params()
                new_params=model.optimizer.prev_operate(params)
                model.set_params(new_params)
                # Conjunto de datos para trabajar
                X_mini_batch,y_mini_batch=mini_batch
                # Construccion del grafo computacional para la funcion de perdida
                s_L=model.build_computational_graph(X_mini_batch, y_mini_batch)
                # Backpropagation
                s_L.backward()
                # Perdida (costo de la funcion de perdida)
                loss=s_L.arr[0,0]
                history_epoch[it_mb]=loss
                # Gradiente de la funcion de perdida
                grad=model.get_grad()
                # Pesos actuales del modelo 
                params=model.get_params()
                # Pesos nuevos del modelo (se utiliza el optimizador)
                new_params=model.optimizer.operate(params, grad)
                # Actualizacion de parametros
                model.set_params(new_params)
                # Se establecen los gradientes a 'None'
                model.reset_grad()

            history[epoch]=history_epoch.mean()

            # Criterio de detencion
            if tol is not None and (epoch > 0 and np.abs(history[epoch] - history[epoch - 1]) < tol):
                stopped=True
                break

        return history,stopped

class ConfusionMatrix():
    def __init__(self, y_pred, y_real, k):
        m=y_real.size
        self.k=k
        self.P=np.zeros(self.k)
        self.R=np.zeros(self.k)
        self.F=np.zeros(self.k)
        self.matrix=np.zeros((self.k,self.k), dtype=int)
        for i in range(m):
            self.matrix[y_real[i,0], y_pred[i,0]]+=1
        for i in range(self.k):
            self.P[i]=0 if self.matrix[:,i].sum() == 0 else self.matrix[i,i] / self.matrix[:,i].sum()
            self.R[i]=0 if self.matrix[i,:].sum() == 0 else self.matrix[i,i] / self.matrix[i,:].sum()
            self.F[i]=0 if (self.P[i] + self.R[i]) == 0 else (2 * self.P[i] * self.R[i]) / (self.P[i] + self.R[i])
        self.precision=self.P.mean()
        self.recall=self.R.mean()
        self.accuracy=np.sum(y_pred == y_real) / m
        self.f1_score=self.F.mean()

    def get_score(self, metric):
        if metric == "precision":
            return self.precision
        if metric == "recall":
            return self.recall
        if metric == "accuracy":
            return self.accuracy
        if metric == "f1_score":
            return self.f1_score

    def __str__(self):
        return f'Precision: {self.precision}\nRecall: {self.recall}\nAccuracy: {self.accuracy}\nF1 Score: {self.f1_score}\n'

class FitResults():
    @staticmethod
    def plot_scores_by_model(scores, model_names, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(model_names, scores)
        for i in range(len(model_names)):
            ax.text(i, scores[i], "{:.2f}".format(scores[i]), ha = 'center')
        ax.set_ylabel('Test set score')
        ax.set_title('Best models comparison')
        return fig,ax

    @staticmethod
    def plot_grid_search_cv(grid_search_cv, figsize=(4,4)):
        comb_names = [f'comb_{comb_name}' for comb_name in grid_search_cv.scores_by_comb_dict.keys()]
        counts = [np.mean(grid_search_cv.scores_by_comb_dict[comb_name]) for comb_name in grid_search_cv.scores_by_comb_dict.keys()]
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(comb_names, counts)
        for i in range(len(comb_names)):
            optimizer_name=grid_search_cv.class_parameters_combinations[i]['optimizer']['name']
            ax.text(i, counts[i], "{:.4f}\n({})".format(counts[i],optimizer_name), ha = 'center')
        ax.set_ylabel('Mean score')
        ax.set_title('Grid Search CV')
        return fig,ax

    @staticmethod
    def plot_history(history, stopped, figsize=(4,4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(history, color='blue')
        if stopped:
            pos=np.where(history == 0)[0][0] - 1
            ax.scatter(pos, history[pos], color='red', label='Stopped')
            ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (training set)')
        ax.set_title('History')
        ax.grid(True)
        return fig,ax

class PlotDatasets():
    @staticmethod
    def plot_2D_classification(X, y, title=None, figsize=(4,4)):
        fig, ax = plt.subplots(figsize=figsize)
        classes=np.unique(y)
        for class_name in classes:
            X_class=X[np.where(y.flatten() == class_name)[0]]
            ax.scatter(X_class[:, 0], X_class[:, 1], c=np.repeat(np.random.rand(1,3), X_class.shape[0], axis=0), label=f'{class_name}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        return fig,ax

    @staticmethod
    def plot_2D_regression(X, y, figsize=(4,4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(X, y, color='blue', label='Points')
        ax.set_xlabel('x1')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()
        return fig,ax

    @staticmethod
    def plot_2D_clusters(X, labels, title=None, figsize=(4,4)):
        fig, ax = plt.subplots(figsize=figsize)
        if labels is not None:
            groups=np.unique(labels)
            for group in groups:
                X_group=X[np.where(labels == group)[0]]
                ax.scatter(X_group[:, 0], X_group[:, 1], c=np.repeat(np.random.rand(1,3), X_group.shape[0], axis=0), label=f'{group}')
        else:
            ax.scatter(X[:, 0], X[:, 1], c='blue', label='Points')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        return fig,ax