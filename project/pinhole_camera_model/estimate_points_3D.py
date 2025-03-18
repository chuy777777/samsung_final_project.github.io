import numpy as np

from pinhole_camera_model.distorted_to_undistorted import DistortedToUndistorted

"""
Clase para calcular puntos 3D a partir de 2 camaras calibradas y de pares de puntos 2D (distorsionados) correspondientes.
"""
class EstimatePoints3D():
    def __init__(self):
        pass

    """
    Calcula puntos 3D.
    
    Entrada: K1 (3,3), K1_inv (3,3), Q1 (3,4), q1 (5,1), K2 (3,3), K2_inv (3,3), Q2 (3,4), q2 (5,1), v3pds1 (N,2), v3pds2 (N,2)
    Salida: vws (N,3)
    """
    @staticmethod
    def estimate_points_3D(K1, K1_inv, Q1, q1, K2, K2_inv, Q2, q2, v3pds1, v3pds2):
        # Numero de puntos 2D
        N=v3pds1.shape[0]
        vws=np.zeros((N,3))
        # Matrices de proyeccion para cada camara
        P1=K1 @ Q1
        P2=K2 @ Q2
        p11_1,p12_1,p13_1,p14_1,p21_1,p22_1,p23_1,p24_1,p31_1,p32_1,p33_1,p34_1=P1.flatten(order='C')
        p11_2,p12_2,p13_2,p14_2,p21_2,p22_2,p23_2,p24_2,p31_2,p32_2,p33_2,p34_2=P2.flatten(order='C')
        # Convertir puntos 2D con distorsion en puntos 2D sin distorsion (para poder utilizar el modelo de camara Pinhole ideal)
        v3ps1=DistortedToUndistorted.undistorted_points(v3pds=v3pds1, K=K1, K_inv=K1_inv, q=q1)
        v3ps2=DistortedToUndistorted.undistorted_points(v3pds=v3pds2, K=K2, K_inv=K2_inv, q=q2)
        # Para cada par de puntos 2D se obtiene su punto 3D correspondiente (en milimetros)
        for n in range(N):
            v3p1=v3ps1[n]
            v3p2=v3ps2[n]
            v3px1,v3py1=v3p1
            v3px2,v3py2=v3p2
            A=np.array([[p31_1 * v3px1 - p11_1,p32_1 * v3px1 - p12_1,p33_1 * v3px1 - p13_1],[p31_1 * v3py1 - p21_1,p32_1 * v3py1 - p22_1,p33_1 * v3py1 - p23_1],[p31_2 * v3px2 - p11_2,p32_2 * v3px2 - p12_2,p33_2 * v3px2 - p13_2],[p31_2 * v3py2 - p21_2,p32_2 * v3py2 - p22_2,p33_2 * v3py2 - p23_2]])
            b=np.array([[p14_1 - p34_1 * v3px1],[p24_1 - p34_1 * v3py1],[p14_2 - p34_2 * v3px2],[p24_2 - p34_2 * v3py2]])
            
            x=np.linalg.inv(A.T @ A) @ A.T @ b
            vw=x.flatten()
            vws[n]=vw

        return vws