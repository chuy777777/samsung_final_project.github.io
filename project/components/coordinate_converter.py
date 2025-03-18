import numpy as np

"""
Clase para convertir puntos de un sistema de coordenadas a otro sistema de coordenadas.
"""
class CoordinateConverter():
    def __init__(self):
        pass

    """
    Convertir puntos del sistema de coordenadas del mundo al sistema de coordenadas tradicional (matplotlib).

    Para un tablero que apunta de frente a la computadora.

    Entrada: vws (m,3)
    Salida: vms (m,3)
    """
    @staticmethod
    def system_w_to_system_m(vws):
        twm=np.zeros((3,1))
        Twm=np.array([[0,0,-1],[-1,0,0],[0,-1,0]])
        H=np.concatenate([np.concatenate([Twm, np.zeros((1,3))], axis=0), np.concatenate([twm, np.ones((1,1))], axis=0)], axis=1)
        vws_=np.concatenate([vws, np.ones((vws.shape[0], 1))], axis=1).T
        vms_=H @ vws_
        # Puntos en forma de vector fila 
        vms=vms_[0:3,:].T
        return vms
