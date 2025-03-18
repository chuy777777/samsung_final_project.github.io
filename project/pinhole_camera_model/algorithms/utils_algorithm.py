import numpy as np

import components.utils as utils

"""
Clase para calcular datos relevantes 
"""
class UtilsAlgorithm():
    def __init__(self):
        pass 

    # Para obtener sistemas de coordenadas asociados a cada punto 3D (cada sistema de coordenadas se calcula a partir de otros puntos 3D)
    def get_coordinate_system_list(self, points_3D, creation_coordinate_system_list):
        # Este es el enfoque que se utiliza para crear los sistemas de coordenadas
        # Los puntos 3D estan en metros y estan en el sistema de coordenadas matplotlib (N,3)
        coordinate_system_list=[]
        number_points=points_3D.shape[0]
        # Para cada punto, se calcula su correspondiente sistema de coordenadas (o matriz de rotacion con respecto al sistema de coordenadas matplotlib)
        for i in range(number_points):
            m,n,p,q=creation_coordinate_system_list[i]
            # Vector en direccion del eje Z
            vz_mn=points_3D[n] - points_3D[m]
            # Vector auxiliar
            v_pq=points_3D[q] - points_3D[p]
            # Vector en direccion del eje Y
            vy=np.cross(v_pq, vz_mn)
            # Vector normalizado en direccion del eje Y
            uy=utils.normalize_vector(vy)
            # Vector normalizado en direccion del eje Z
            uz_mn=utils.normalize_vector(vz_mn)
            # Vector normalizado en direccion del eje X
            ux=np.cross(uy, uz_mn)
            # Matriz de rotacion (sistema de coordenadas)
            R=np.concatenate([ux[:,None],uy[:,None],uz_mn[:,None]], axis=1)
            coordinate_system_list.append(R)

        return coordinate_system_list
    
    # Para calcular los angulos de Euler de sistemas de coordenadas
    def get_euler_angles(self, coordinate_system_list):
        N=len(coordinate_system_list)
        euler_angles=np.zeros((N,3))
        for n in range(N):
            euler_angles[n]=utils.euler_angles_from_to_rotation_matrix(R=coordinate_system_list[n])
        return euler_angles
    
    # Para calcular los angulos de Euler para clasificacion
    def get_classification_euler_angles(self, coordinate_system_list):
        pq_coordinate_systems=[
            # Izquierda
            (25,27),(23,25),(11,23),(11,13),(13,15),(23,11),(11,0),
            # Derecha
            (26,28),(24,26),(12,24),(12,14),(14,16),(24,12),(12,0)
        ]
        classification_euler_angles=np.zeros((len(pq_coordinate_systems),3))
        for i in range(len(pq_coordinate_systems)):
            p,q=pq_coordinate_systems[i]
            Rpm=coordinate_system_list[p]
            Rqm=coordinate_system_list[q]
            euler_angles=None
            try:
                Rqp=np.linalg.inv(Rpm) @ Rqm
                euler_angles=utils.euler_angles_from_to_rotation_matrix(Rqp)
            except np.linalg.LinAlgError:
                euler_angles=np.zeros(3)
            classification_euler_angles[i]=euler_angles
        return classification_euler_angles
    
    # Para calcular datos relevantes a partir de puntos 3D
    def get_data_from_points_3D(self, points_3D, creation_coordinate_system_list):
        coordinate_system_list=self.get_coordinate_system_list(points_3D=points_3D, creation_coordinate_system_list=creation_coordinate_system_list)
        euler_angles=self.get_euler_angles(coordinate_system_list=coordinate_system_list)
        classification_euler_angles=self.get_classification_euler_angles(coordinate_system_list=coordinate_system_list)
        return euler_angles,classification_euler_angles,coordinate_system_list
    
    