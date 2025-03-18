import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.coordinate_converter import CoordinateConverter
from pinhole_camera_model.estimate_points_3D import EstimatePoints3D
from pinhole_camera_model.draw_image import DrawImage

# ALGORITMOS
from pinhole_camera_model.algorithms.algorithm_mediapipe_pose import AlgorithmMediaPipePose

"""
Frame para la seleccion del algoritmo de deteccion de puntos 2D caracteristicos del cuerpo
humano sobre una imagen.
"""
class FrameSelectAlgorithm(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        self.factor_mm_to_mt=1 / 1000
        self.algorithm=None
        self.algorithm_class=AlgorithmMediaPipePose

        label_algorithm=ctk.CTkLabel(master=self, text="Seleccion del algoritmo")
        self.var_selected_algorithm=ctk.IntVar(value=0)
        check_box_algorithm=ctk.CTkCheckBox(master=self, text=self.algorithm_class.algorithm_name, variable=self.var_selected_algorithm, onvalue=1, offvalue=0)
        self.insert_element(cad_pos="0,0", element=label_algorithm, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=check_box_algorithm, padx=5, pady=5, sticky="")

    """
    Con respecto a los puntos 3D calculados por el modelo de camara Pinhole:
        - Se asume que el sistema de coordenadas del mundo queda de tal manera que el tablero
            esta apuntando de frente a las camaras
        - Los puntos 3D se pasan al systema de coordenadas de matplotlib (tradicional)
        - Los puntos se convierten de mm a mt
    """
    def set_estimation_data_from_algorithm(self, Q1, Q2, frame_bgr_1=None, frame_bgr_2=None, draw=True):
        # Se verifica si el algoritmo esta seleccionado
        if self.var_selected_algorithm.get():
            if self.algorithm is None:
                self.algorithm=self.algorithm_class()
        else:
            self.algorithm=None
        # Se obtienen las imagenes de cada camara
        frame_bgr_1=self.thread_camera_1.frame_bgr if frame_bgr_1 is None else frame_bgr_1
        frame_bgr_2=self.thread_camera_2.frame_bgr if frame_bgr_2 is None else frame_bgr_2
        # Se obtienen los parametros de calibracion de cada camara
        camera_device_1=self.thread_camera_1.camera_device if self.thread_camera_1.camera_device.calibration_information_is_loaded else None
        camera_device_2=self.thread_camera_2.camera_device if self.thread_camera_2.camera_device.calibration_information_is_loaded else None
        # Calculamos datos solo si esta el algoritmo seleccionado
        if self.algorithm is not None:
            # Se optienen los puntos 2D detectados por el algoritmo para cada camara
            v3pds1,is_ok1=self.algorithm.get_points_2D(frame_bgr=frame_bgr_1)
            v3pds2,is_ok2=self.algorithm.get_points_2D(frame_bgr=frame_bgr_2)
            # En caso de especificarse, se dibujan los puntos 2D detectadso en la imagen de cada camara
            if draw:
                if is_ok1:
                    frame_bgr_1=DrawImage.draw_conenctions(frame_bgr=frame_bgr_1, points=v3pds1, connection_list=self.algorithm.connection_list)
                if is_ok2:
                    frame_bgr_2=DrawImage.draw_conenctions(frame_bgr=frame_bgr_2, points=v3pds2, connection_list=self.algorithm.connection_list)
            # Los puntos 2D detectados en cada camara, ambas camaras calibradas y las matrices extrinsecas (todo debe estar bien)
            if is_ok1 and is_ok2 and Q1 is not None and Q2 is not None and camera_device_1 is not None and camera_device_2 is not None:
                # Se calculan los puntos 3D en el sistema de coordenadas del mundo de cada par de puntos correspondientes 2D
                vws=EstimatePoints3D.estimate_points_3D(K1=camera_device_1.K, K1_inv=camera_device_1.K_inv, Q1=Q1, q1=camera_device_1.q, K2=camera_device_2.K, K2_inv=camera_device_2.K_inv, Q2=Q2, q2=camera_device_2.q, v3pds1=v3pds1, v3pds2=v3pds2)
                # Los puntos 3D se transforman al sistema de coordenadas tradicional (matplotlib) y se pasan las unidades de milimetros a metros
                vms=CoordinateConverter.system_w_to_system_m(vws=vws) * self.factor_mm_to_mt
                # Se guardan los puntos 3D
                self.algorithm.set_data(points_3D=vms)
            else:
                self.algorithm.set_data(points_3D=None)
                    
        # Se devuelven las imagenes de cada camara
        return frame_bgr_1,frame_bgr_2