import numpy as np

class KMeans():
    def __init__(
            self, 
            k=3, 
            distance_measure="euclidean"
        ):
        # Se establecen los parametros de clase (hiperparametros)
        self.set_class_parameters(k, distance_measure)
        # Para guardar el conjunto de datos
        self.X=None
        # Para guardar los centroides de cada cluster
        self.centroids=None
        # Para guardar las etiquetas de cluster de cada muestra del conjunto de datos
        self.labels=None
        self.name='kmeans'
        
    def set_class_parameters(self, k, distance_measure):
        # Numero de clusteres
        self.k=k
        # Medida de distancia ["euclidean", "manhattan"]
        self.distance_measure=distance_measure

    def distances_to_centroids(self, p):
        # Se calcula la distancia entre la muestra y cada centroide
        distances=None
        if self.distance_measure == 'euclidean':
            distances=np.linalg.norm(self.centroids - p, axis=1)
        if self.distance_measure == 'manhattan':
            distances=np.sum(np.abs(self.centroids - p), axis=1)
        return distances
        
    def fit(self, X, params_init, it=100, tol=1e-4):
        m,n=X.shape
        self.X=X
        self.labels=np.zeros(m)
        # Inicializacion de los centroides
        params_init(X, self)
        # Para cada iteracion
        for i in range(it):
            # Para cada muestra
            for j in range(m):
                pj=X[j]
                # Se le asigna el centroide mas cercano
                self.labels[j]=np.argmin(self.distances_to_centroids(pj))
            new_centroids=np.zeros((self.k,n))
            # Se recalcula cada centroide
            for j in range(self.k):
                X_j=X[self.labels == j]
                if X_j.shape[0] != 0:
                    new_centroids[j]=(1 / X_j.shape[0]) * np.sum(X_j, axis=0)
                else:
                    # Si ningun dato fue asignado a dicho centroide, entonces se mueve de posicion dicho centroide (a algun punto aleatorio)
                    new_centroids[j]=X[np.random.randint(0,m)]
            # Si los cambios entre centroides es muy pequeño, entonces se finaliza
            if np.mean(np.linalg.norm(new_centroids - self.centroids, axis=0)) < tol:
                break
            # Se actualizan los centroides
            self.centroids=new_centroids
        self.labels=self.labels.astype(int)       
        
    def predict(self, X):
        # Prediccion del grupo al que pertenecen
        m,n=X.shape
        labels=np.zeros(m)
        # Para cada muestra
        for j in range(m):
            pj=X[j]
            # Se le asigna el centroide mas cercano
            self.labels[j]=np.argmin(self.distances_to_centroids(pj))
        labels=labels.astype(int)   
        return labels
        
    def score(self, metric="dunn_index"):
        # Mide la calidad de las agrupaciones
        if metric == 'dunn_index':
            comb=[]
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    comb.append((i,j))
            inter_cluster_distances=[]
            for i,j in comb:
                d=np.linalg.norm(self.centroids[i] - self.centroids[j])
                inter_cluster_distances.append(d)
            intra_cluster_distances=[]
            for i in range(self.k):
                X_cluster=self.X[self.labels == i]
                d_min=np.min(np.linalg.norm(X_cluster - self.centroids[i], axis=1))
                intra_cluster_distances.append(d_min)
            dunn_index=np.min(inter_cluster_distances) / np.max(intra_cluster_distances)
            return dunn_index
            