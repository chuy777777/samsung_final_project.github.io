import customtkinter  as ctk
import tkinter as tk
import numpy as np

from components.create_frame import CreateFrame
from components.create_scrollable_frame import CreateScrollableFrame
from components.grid_frame import GridFrame
from components.frame_camera_display import FrameCameraDisplay
from pages.parameter_checking.frame_calibration_information import FrameCalibrationInformation
from pages.estimation.frame_estimation_graphic_3D_with_options import FrameEstimationGraphic3DWithOptions
from pages.estimation.frame_calculate_extrinsic_matrices import FrameCalculateExtrinsicMatrices
from pages.estimation.frame_select_algorithm import FrameSelectAlgorithm
from pages.estimation.frame_save_estimations import FrameSaveEstimations

"""
Frame para mostrar las estimaciones 3D.
"""
class FrameEstimation(CreateScrollableFrame):
    def __init__(self, master, name, callback_estimations=None, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(4,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        self.callback_estimations=callback_estimations
        self.rate_ms=50

        self.frame_select_algorithm=FrameSelectAlgorithm(master=self, name="FrameSelectAlgorithm")

        # Mostrar informacion de la calibracion de las camaras
        frame_calibration_informations=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        frame_calibration_information_camera_1=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation1", thread_camera=self.thread_camera_1, only_camera_parameters=True)
        frame_calibration_information_camera_2=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation2", thread_camera=self.thread_camera_2, only_camera_parameters=True)
        frame_calibration_informations.insert_element(cad_pos="0,0", element=frame_calibration_information_camera_1, padx=5, pady=5, sticky="")
        frame_calibration_informations.insert_element(cad_pos="0,1", element=frame_calibration_information_camera_2, padx=5, pady=5, sticky="")

        # Visualizacion de las camaras
        frame_camera_displays=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,2), arr=np.array([["0,0","0,0"],["1,0","1,0"],["2,0","2,1"]])))
        self.var_draw_algorithms=ctk.IntVar(value=1)
        check_box_draw_algorithms=ctk.CTkCheckBox(master=frame_camera_displays, text="Mostrar algoritmos en imagen", variable=self.var_draw_algorithms, onvalue=1, offvalue=0)
        # Este frame nos ayuda a construir el conjunto de datos
        self.frame_save_estimations=FrameSaveEstimations(master=frame_camera_displays, name="FrameSaveEstimations")
        self.frame_camera_display_camera_1=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay1", thread_camera=self.thread_camera_1, rate_ms=self.rate_ms, scale_percent=70, editable=True)
        self.frame_camera_display_camera_2=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay2", thread_camera=self.thread_camera_2, rate_ms=self.rate_ms, scale_percent=70, editable=True)
        frame_camera_displays.insert_element(cad_pos="0,0", element=check_box_draw_algorithms, padx=5, pady=5, sticky="")
        frame_camera_displays.insert_element(cad_pos="1,0", element=self.frame_save_estimations, padx=5, pady=5, sticky="")
        frame_camera_displays.insert_element(cad_pos="2,0", element=self.frame_camera_display_camera_1, padx=5, pady=5, sticky="")
        frame_camera_displays.insert_element(cad_pos="2,1", element=self.frame_camera_display_camera_2, padx=5, pady=5, sticky="")

        # Grafico 3D y calculo de matrices extrinsecas
        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        self.frame_estimation_graphic_3D_with_options=FrameEstimationGraphic3DWithOptions(master=frame_container, name="FrameEstimationGraphic3DWithOptions", show_select_angles=True, square_size=1, width=400, height=400)
        self.frame_calculate_extrinsic_matrices=FrameCalculateExtrinsicMatrices(master=frame_container, name="FrameCalculateExtrinsicMatrices")
        frame_container.insert_element(cad_pos="0,0", element=self.frame_estimation_graphic_3D_with_options, padx=5, pady=5, sticky="n")
        frame_container.insert_element(cad_pos="0,1", element=self.frame_calculate_extrinsic_matrices, padx=5, pady=5, sticky="n")

        self.insert_element(cad_pos="0,0", element=self.frame_select_algorithm, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=frame_calibration_informations, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=frame_camera_displays, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=frame_container, padx=5, pady=5, sticky="new")

        self.start_process()

    # Para comenzar el proceso de mostrar resultados de las estimaciones en cada camara (se repite cada cierto tiempo)
    def start_process(self):
        # Se obtienen las matrices extrinsecas de cada camara (en caso de que se hayan calculado)
        Q1=self.frame_calculate_extrinsic_matrices.dict_extrinsic_matrices["Q1"]
        Q2=self.frame_calculate_extrinsic_matrices.dict_extrinsic_matrices["Q2"]
        # Se realizan los calculos de informacion relevante (estimaciones 3D, ...)
        frame_bgr_1,frame_bgr_2=self.frame_select_algorithm.set_estimation_data_from_algorithm(Q1=Q1, Q2=Q2, draw=self.var_draw_algorithms.get())
        # Se muestran las imagenes de cada camara
        self.frame_camera_display_camera_1.update_label_camera(frame_bgr=frame_bgr_1)
        self.frame_camera_display_camera_2.update_label_camera(frame_bgr=frame_bgr_2)
        # Se dibuja informacion relevante
        self.frame_estimation_graphic_3D_with_options.draw_estimation(algorithm=self.frame_select_algorithm.algorithm)
        # Se le pasan los algoritmos al frame que guarda las estimaciones 3D 
        self.frame_save_estimations.set_algorithm(algorithm=self.frame_select_algorithm.algorithm)
        # Se le pasan los algoritmos al 'callback'
        if self.callback_estimations is not None:
            self.callback_estimations(algorithm=self.frame_select_algorithm.algorithm)

        self.after(self.rate_ms, self.start_process)
        