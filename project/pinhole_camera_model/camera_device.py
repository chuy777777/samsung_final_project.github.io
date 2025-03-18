import numpy as np
import os
import copy

from pinhole_camera_model.calibration_parameters import CalibrationParameters
from pinhole_camera_model.optimization_parameters import OptimizationParameters

"""
Clase para guardar los parametros de la camara, asi como datos importantes de la camara.
"""
class CameraDevice():
    def __init__(self):
        self.init_device()

    # Para crear una copia del objeto
    def copy(self):
        return copy.deepcopy(self)

    # Para inicializar las variables de la clase
    def init_device(self):
        self.camera_name=""
        self.camera_path=""
        self.camera_information_is_loaded=False
        self.init_calibration_information()

    # Para inicializar las variables del proceso de calibracion
    def init_calibration_information(self):
        self.K=None
        self.K_inv=None
        self.q=None
        self.Qs=None
        self.lambdas=None
        self.history_L=None
        self.history_norm_grad=None
        self.calibration_parameters=None
        self.optimization_parameters=None
        self.calibration_information_is_loaded=False
        
    # Para establecer el nombre y la ruta de la camara
    def set_device(self, camera_name, camera_path):
        self.camera_name=camera_name
        self.camera_path=camera_path
        self.camera_information_is_loaded=True

    # Para cargar nueva informacion de calibracion 
    def load_new_calibration_information(self, K, K_inv, q, Qs, lambdas, history_L, history_norm_grad, calibration_parameters, optimization_parameters):
        if self.camera_information_is_loaded:
            self.K=K 
            self.K_inv=K_inv
            self.q=q
            self.Qs=Qs
            self.lambdas=lambdas
            self.history_L=history_L
            self.history_norm_grad=history_norm_grad
            self.calibration_parameters=calibration_parameters
            self.optimization_parameters=optimization_parameters
            self.calibration_information_is_loaded=True

    # Para cargar informacion de calibracion previamente guardada 
    def load_saved_calibration_information(self, folder_name):
        if self.camera_information_is_loaded:
            full_path=os.path.join(folder_name, *[self.camera_name])
            if os.path.exists(full_path):
                self.K=np.load(os.path.join(full_path, *["K.npy"])) 
                self.K_inv=np.load(os.path.join(full_path, *["K_inv.npy"])) 
                self.q=np.load(os.path.join(full_path, *["q.npy"])) 
                self.Qs=np.load(os.path.join(full_path, *["Qs.npy"])) 
                self.lambdas=np.load(os.path.join(full_path, *["lambdas.npy"])) 
                self.history_L=np.load(os.path.join(full_path, *["history_L.npy"])) 
                self.history_norm_grad=np.load(os.path.join(full_path, *["history_norm_grad.npy"]))
                self.calibration_parameters=CalibrationParameters.load(full_path=full_path)
                self.optimization_parameters=OptimizationParameters.load(full_path=full_path)
                self.calibration_information_is_loaded=True
            else:
                self.init_calibration_information()

    # Para guardar la informacion de calibracion en la computadora
    def save_calibration_information(self, folder_name):
        if self.camera_information_is_loaded:
            if self.calibration_information_is_loaded:
                full_path=os.path.join(folder_name, *[self.camera_name])
                if not os.path.exists(full_path):
                    os.mkdir(full_path)
                np.save(os.path.join(full_path, *["K.npy"]), self.K) 
                np.save(os.path.join(full_path, *["K_inv.npy"]), self.K_inv) 
                np.save(os.path.join(full_path, *["q.npy"]), self.q) 
                np.save(os.path.join(full_path, *["Qs.npy"]), self.Qs) 
                np.save(os.path.join(full_path, *["lambdas.npy"]), self.lambdas) 
                np.save(os.path.join(full_path, *["history_L.npy"]), self.history_L) 
                np.save(os.path.join(full_path, *["history_norm_grad.npy"]), self.history_norm_grad) 
                self.calibration_parameters.save(full_path=full_path)
                self.optimization_parameters.save(full_path=full_path)

    # Para mostrar todos los valores de las variables del proceso de calibracion
    def __str__(self):
        cad="{}\n\n".format(self.camera_name.upper())
        cad+="PARAMETROS DE LA CAMARA\n\n"
        cad+="K:\n{}\n\n".format(np.array2string(self.K))
        cad+="q:\n{}\n\n".format(np.array2string(self.q))
        cad+="PARAMETROS DE CALIBRACION\n\n"
        cad+="Dimensiones tablero: {}\n".format(self.calibration_parameters.chessboard_dimensions)
        cad+="Tamaño cuadrado: {}\n".format(self.calibration_parameters.square_size)
        return cad