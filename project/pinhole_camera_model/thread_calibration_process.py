import numpy as np
from threading import Thread

from pinhole_camera_model.camera_calibration import CameraCalibration

"""
Clase que ejecuta un hilo (thread) para el proceso de calibracion.
"""
class ThreadCalibrationProcess(Thread):
    def __init__(self, thread_camera, v3pds_list, vws, calibration_parameters, optimization_parameters):
        Thread.__init__(self, daemon=True)
        # Hilo que ejecuta una camara
        self.thread_camera=thread_camera
        # Lista de puntos 2D (distorsionados) de cada imagen tomada del tablero de ajedrez ([(N,2), ...])
        self.v3pds_list=v3pds_list
        # Puntos 3D del tablero de ajedrez (N,3)
        self.vws=vws
        # Parametros de calibracion
        self.calibration_parameters=calibration_parameters
        # Parametros de optimizacion
        self.optimization_parameters=optimization_parameters
        # Para indicar si todo ha salido bien en el proceso de calibracion
        self.is_ok=False

    # Para iniciar el hilo
    def run(self):
        self.loop_calibration_process()
        print("End ThreadCalibrationProcess ({})".format(self.thread_camera.camera_device.camera_name))    

    # Funcion que ejecuta el hilo
    def loop_calibration_process(self):
        # Se calculan los parametros optimos de la camara
        K_opt,q_opt,Qs_opt,lambdas_opt,history_L,history_norm_grad,is_ok=CameraCalibration.calculate_optimal_parameters(v3pds_list=self.v3pds_list, vws=self.vws, num_it=self.optimization_parameters.num_it, lr=self.optimization_parameters.lr, beta_1=self.optimization_parameters.beta_1, beta_2=self.optimization_parameters.beta_2)
        if is_ok:
            # Cargar y guardar parametros
            self.thread_camera.load_new_calibration_information(K=K_opt, K_inv=np.linalg.inv(K_opt), q=q_opt, Qs=Qs_opt, lambdas=lambdas_opt, history_L=history_L, history_norm_grad=history_norm_grad, calibration_parameters=self.calibration_parameters, optimization_parameters=self.optimization_parameters)
            self.thread_camera.save_calibration_information()
        else:
            """
            Ocurrio algun problema:
                - Debido a la descomposicion de Cholesky
                    La matriz B no logro ser una matriz simetrica definida positiva
                - Debido a otra excepcion
            """
        self.is_ok=is_ok
        