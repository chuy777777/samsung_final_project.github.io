import numpy as np
from scipy.optimize import root

"""
Clase para transformar puntos 2D distorsionados en puntos 2D sin distorsion.
"""
class DistortedToUndistorted():
    def __init__(self):
        pass

    """
    Funcion para determinar su raiz (valor de 'v1n' que hace que la funcion sea 0).
    
    Entrada: v1n (2,), v3pd (2,), K_inv (3,3), q (5,1)
    Salida: val_f (2,)
    """
    @staticmethod
    def f(v1n, v3pd, K_inv, q):
        v1nd=DistortedToUndistorted.get_v1nd(v3pd=v3pd, K_inv=K_inv)
        k1,k2,k3,p1,p2=q.flatten()
        v1nx,v1ny=v1n
        r=np.linalg.norm(v1n)
        d=np.array([v1nx * (k1 * r**2 + k2 * r**4 + k3 * r**6) + 2 * p1 * v1nx * v1ny + p2 * (r**2 + 2 * v1nx**2),v1ny * (k1 * r**2 + k2 * r**4 + k3 * r**6) + p1 * (r**2 + 2 * v1ny**2)+ 2 * p2 * v1nx * v1ny ])
        val_f=v1nd - v1n - d

        return val_f

    """
    Devuelve 'v1nd'.
    
    Entrada: v3pd (2,), K_inv (3,3)
    Salida: v1nd (2,)
    """
    @staticmethod
    def get_v1nd(v3pd, K_inv):
        md=np.concatenate([v3pd, np.ones((1))], axis=0)[:,None]
        v1nd=(K_inv @ md)[0:2].flatten()

        return v1nd

    """
    Devuelve los puntos 2D sin distorsion.
    
    Entrada: v3pds (N,2), K (3,3), K_inv (3,3), q (5,1)
    Salida: v3ps (N,2)
    """
    @staticmethod
    def undistorted_points(v3pds, K, K_inv, q, show_progress=False):
        v3ps=np.zeros(v3pds.shape)
        N=v3ps.shape[0]
        for n in range(N):
            if show_progress:
                print("{} from {}".format(n + 1,N))
            v3pd=v3pds[n]
            initial_v1n=DistortedToUndistorted.get_v1nd(v3pd=v3pd, K_inv=K_inv)
            res=root(lambda v1n: DistortedToUndistorted.f(v1n=v1n, v3pd=v3pd, K_inv=K_inv, q=q), initial_v1n, method='hybr', jac=False)
            v1n=res.x
            v3p=(K @ np.concatenate([v1n, np.ones((1))], axis=0)[:,None])[0:2]
            v3ps[n]=v3p.flatten()

        return v3ps
    
    """
    Devuelve una imagen sin distorsion.
    
    Entrada: distorted_frame_bgr (frame de la camara con distorsion), K (3,3), K_inv (3,3), q (5,1)
    Salida: undistorted_frame_bgr (frame de la camara sin distorsion)
    """
    def undistorted_image(distorted_frame_bgr, K, K_inv, q, show_progress=True):
        undistorted_frame_bgr=np.zeros(distorted_frame_bgr.shape, dtype="uint8")
        h,w,d=undistorted_frame_bgr.shape
        # De izquierda a derecha de arriba a abajo
        indexes=list(np.ndindex((h,w)))
        N=len(indexes)
        v3pds=np.zeros((N,2))
        for n in range(N):
            v3pds[n]=np.array([indexes[n][1],indexes[n][0]])
        v3ps=DistortedToUndistorted.undistorted_points(v3pds=v3pds, K=K, K_inv=K_inv, q=q, show_progress=show_progress)
        v3ps=v3ps.astype(int)
        for n in range(N):
            v3px,v3py=v3ps[n]
            pos_y,pos_x=indexes[n]
            if (v3px >= 0 and v3px < w) and (v3py >= 0 and v3py < h):
                undistorted_frame_bgr[pos_y,pos_x,:]=distorted_frame_bgr[v3py,v3px,:]

        return undistorted_frame_bgr