import cv2
import time
from threading import Thread

from components.template_thread import TemplateThread
from pinhole_camera_model.camera_device import CameraDevice

"""
Clase de la que heredan las clases que requieren ser notificados cuando existan cambios en las camaras.
"""
class ThreadCameraFrameListener():
    def __init__(self):
        pass 

    # Sobreescribir este metodo
    def thread_camera_frame_listener_notification(self):
        pass

"""
Clase que ejecuta un hilo (thread) que accede a lo que captura una camara.

La clase 'CameraDevice' es la que almacena toda la informacion del proceso de calibracion.
"""
class ThreadCamera(Thread, TemplateThread):
    def __init__(self, root_name, folder_name_optimal_parameters):
        Thread.__init__(self, daemon=True)
        TemplateThread.__init__(self)
        self.root_name=root_name
        self.folder_name_optimal_parameters=folder_name_optimal_parameters
        self.camera_device=CameraDevice()
        self.cap=None
        self.frame_bgr=None
        self.frame_bgr_shape=None
        self.frame_listener_list=[]

    # Para iniciar el hilo
    def run(self):
        self.init_cap()
        self.view_camera()       
        print("End ThreadCamera from '{}'".format(self.camera_device.camera_name))    

    # Para inicializar la captura de video
    def init_cap(self, camera_name="", camera_path=""):
        self.release_cap()
        # Cargar datos de la camara
        if camera_name != "":
            self.set_device_and_load_saved_calibration_information(camera_name=camera_name, camera_path=camera_path)
        else:
            self.camera_device.init_device()
        # Crear captura de video
        if camera_path != "":
            self.cap=cv2.VideoCapture(camera_path)
        else:
            self.cap=cv2.VideoCapture() 
        # Notificar que se han realizado cambios en la camara
        self.notify_frame_listeners()
        
    # Para cerrar la captura de video
    def release_cap(self):
        if self.cap is not None:
            self.cap.release()

    # Para guardar cada frame de lo que captura la camara (se repite constantemente este proceso)
    def view_camera(self):
        while not self.event_kill_thread.is_set():
            try:
                if self.cap is not None:
                    if self.cap.isOpened():
                        ret, frame_bgr=self.cap.read()
                        if ret:
                            # Se guarda el frame y su tamaño
                            self.frame_bgr=frame_bgr
                            self.frame_bgr_shape=frame_bgr.shape
                        else:
                            """
                            Una vez que esta funcionando, si la camara se desconecta entra aqui.
                            Si se vuelve a conectar por USB no se reconecta por si sola.
                            """
                            self.frame_bgr=None
                            self.frame_bgr_shape=None
                    else:
                        """
                        Si la camara no esta conectada por USB desde un inicio entra aqui.
                        Si se vuelve a conectar por USB no se reconecta por si sola.
                        """
                        self.frame_bgr=None
                        self.frame_bgr_shape=None
                else:
                    self.frame_bgr=None
                    self.frame_bgr_shape=None
            except cv2.error as e:
                print("(cv2.error): {}".format(e))
                self.frame_bgr=None
                self.frame_bgr_shape=None

            time.sleep(0.001)
        self.release_cap()
        
    # Para establecer y cargar los datos de la camara 
    def set_device_and_load_saved_calibration_information(self, camera_name, camera_path):
        self.camera_device.set_device(camera_name=camera_name, camera_path=camera_path)
        self.camera_device.load_saved_calibration_information(folder_name=self.folder_name_optimal_parameters)
    
    # Para cargar nuevos datos a la camara y notificar los cambios
    def load_new_calibration_information(self, K, K_inv, q, Qs, lambdas, history_L, history_norm_grad, calibration_parameters, optimization_parameters):
        self.camera_device.load_new_calibration_information(K=K, K_inv=K_inv, q=q, Qs=Qs, lambdas=lambdas, history_L=history_L, history_norm_grad=history_norm_grad, calibration_parameters=calibration_parameters, optimization_parameters=optimization_parameters)
        self.notify_frame_listeners()
        
    # Para guardar los datos de la camara
    def save_calibration_information(self):
        self.camera_device.save_calibration_information(folder_name=self.folder_name_optimal_parameters)
    
    # Para agregar a un oyente de los cambios en la camara
    def add_frame_listener(self, frame_listener):
        frame_listener_name_list=list(map(lambda elem: elem.name, self.frame_listener_list))
        if frame_listener.name not in frame_listener_name_list:
            self.frame_listener_list.append(frame_listener)

    # Para eliminar a un oyente de los cambios en la camara
    def delete_frame_listener(self, frame_listener_name):
        self.frame_listener_list=list(filter(lambda elem: elem.name != frame_listener_name,self.frame_listener_list))

    # Para notificar los cambios de la camara
    def notify_frame_listeners(self):
        for frame_listener in self.frame_listener_list:
            frame_listener.thread_camera_frame_listener_notification()