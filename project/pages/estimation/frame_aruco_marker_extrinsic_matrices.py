import customtkinter  as ctk

from components.text_validator import TextValidator
from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from pinhole_camera_model.image_points_2D_3D import ImagePoints2D3D, dict_aruco
from pinhole_camera_model.camera_calibration import CameraCalibration

"""
Clase para las opciones de configuracion del marcador ArUco.
"""
class ArUcoMarkerPatternEstimateQ():
    def __init__(self, aruco_type, aruco_id, square_size):
        self.aruco_type=aruco_type
        self.aruco_id=aruco_id
        self.square_size=square_size

    @staticmethod
    def validate(aruco_type, text_aruco_id, text_square_size):
        n=int(aruco_type.split("_")[2])
        aruco_id=TextValidator.validate_number(text=text_aruco_id)
        square_size=TextValidator.validate_number(text=text_square_size)
        if aruco_id is not None and square_size is not None:
            if aruco_id < n and square_size > 0:
                return ArUcoMarkerPatternEstimateQ(aruco_type=aruco_type, aruco_id=aruco_id, square_size=square_size)
            else:
                return None
        else:
            return None

"""
Frame para configurar las opciones del marcador ArUco para el calculo de las matrices extrinsecas de las camaras.
"""
class FrameArUcoMarkerExtrinsicMatrices(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(6,1), arr=None), **kwargs)
        
        self.var_aruco_type=ctk.StringVar(value="DICT_7X7_50")
        self.var_aruco_id=ctk.StringVar(value="0")
        self.var_square_size=ctk.StringVar(value="145")
        label_title=ctk.CTkLabel(master=self, text="Configuraciones para el calculo de las matrices extrinsecas")
        option_menu_aruco_type=ctk.CTkOptionMenu(master=self, values=list(dict_aruco.keys()), variable=self.var_aruco_type)
        label_aruco_id=ctk.CTkLabel(master=self, text="ID aruco")
        entry_aruco_id=ctk.CTkEntry(master=self, textvariable=self.var_aruco_id)
        label_square_size=ctk.CTkLabel(master=self, text="Tamaño cuadrado (mm)")
        entry_square_size=ctk.CTkEntry(master=self, textvariable=self.var_square_size)

        self.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=option_menu_aruco_type, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=label_aruco_id, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=entry_aruco_id, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="4,0", element=label_square_size, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="5,0", element=entry_square_size, padx=5, pady=5, sticky="")
    
    # Para calcular las matrices extrinsecas de las camaras
    def get_extrinsic_matrices(self, thread_camera_1, thread_camera_2):
        arucoMarkerPatternEstimateQ=ArUcoMarkerPatternEstimateQ.validate(aruco_type=self.var_aruco_type.get(), text_aruco_id=self.var_aruco_id.get(), text_square_size=self.var_square_size.get())
        if arucoMarkerPatternEstimateQ is not None:
            Q1,Q2=None,None
            v3pds,vws,is_ok=ImagePoints2D3D.point_pairs_2D_and_3D_aruco_marker(frame_bgr=thread_camera_1.frame_bgr, aruco_type=arucoMarkerPatternEstimateQ.aruco_type, aruco_id=arucoMarkerPatternEstimateQ.aruco_id, square_size=arucoMarkerPatternEstimateQ.square_size)
            if is_ok:
                Q1=CameraCalibration.get_Q(v3pds=v3pds, vws=vws, K=thread_camera_1.camera_device.K, K_inv=thread_camera_1.camera_device.K_inv, q=thread_camera_1.camera_device.q)
            v3pds,vws,is_ok=ImagePoints2D3D.point_pairs_2D_and_3D_aruco_marker(frame_bgr=thread_camera_2.frame_bgr, aruco_type=arucoMarkerPatternEstimateQ.aruco_type, aruco_id=arucoMarkerPatternEstimateQ.aruco_id, square_size=arucoMarkerPatternEstimateQ.square_size)
            if is_ok:
                Q2=CameraCalibration.get_Q(v3pds=v3pds, vws=vws, K=thread_camera_2.camera_device.K, K_inv=thread_camera_2.camera_device.K_inv, q=thread_camera_2.camera_device.q)
            if Q1 is not None and Q2 is not None:
                return Q1,Q2,True 
            else: 
                return None,None,False
        else:
            return None,None,False