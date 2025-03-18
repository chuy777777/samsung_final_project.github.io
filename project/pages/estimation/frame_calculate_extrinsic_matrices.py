import customtkinter  as ctk
import tkinter as tk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from pages.estimation.frame_aruco_marker_extrinsic_matrices import FrameArUcoMarkerExtrinsicMatrices

"""
Frame para calcular las matrices extrinsecas de las camaras.

Guarda las matrices extrinsecas de las camaras.
"""
class FrameCalculateExtrinsicMatrices(CreateFrame):
    def __init__(self, master, name, callback=None, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        self.callback=callback
        self.dict_extrinsic_matrices={"Q1": None, "Q2": None}

        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        self.frame_aruco_marker_extrinsic_matrices=FrameArUcoMarkerExtrinsicMatrices(master=frame_container, name="FrameArUcoMarkerExtrinsicMatrices")
        frame_container_estimation=CreateFrame(master=frame_container, grid_frame=GridFrame(dim=(3,1), arr=None))
        button_calculate_extrinsic_matrices=ctk.CTkButton(master=frame_container_estimation, text="Calcular matrices extrinsecas", fg_color="green2", hover_color="green3", command=self.calculate_extrinsic_matrices)
        button_delete_extrinsic_matrices=ctk.CTkButton(master=frame_container_estimation, text="Eliminar matrices extrinsecas", fg_color="red2", hover_color="red3", command=self.delete_extrinsic_matrices)
        self.label_status=ctk.CTkLabel(master=frame_container_estimation, text="", width=100, height=100)
        frame_container_estimation.insert_element(cad_pos="0,0", element=button_calculate_extrinsic_matrices, padx=5, pady=5, sticky="")
        frame_container_estimation.insert_element(cad_pos="1,0", element=button_delete_extrinsic_matrices, padx=5, pady=5, sticky="")
        frame_container_estimation.insert_element(cad_pos="2,0", element=self.label_status, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="0,0", element=frame_container_estimation, padx=5, pady=5, sticky="n")
        frame_container.insert_element(cad_pos="1,0", element=self.frame_aruco_marker_extrinsic_matrices, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=frame_container, padx=5, pady=5, sticky="new")

    # Para guardar las matrices extrinsecas de las camaras.
    def set_extrinsic_matrices(self, Q1=None, Q2=None):
        self.label_status.configure(bg_color="red" if Q1 is None and Q2 is None else "green")
        self.dict_extrinsic_matrices={"Q1": Q1, "Q2": Q2}
        if self.callback is not None:
            self.callback()

    # Para calcular las matrices extrinsecas de las camaras.
    def calculate_extrinsic_matrices(self):
        if self.thread_camera_1.camera_device.calibration_information_is_loaded and self.thread_camera_2.camera_device.calibration_information_is_loaded:
            Q1,Q2,is_ok=self.frame_aruco_marker_extrinsic_matrices.get_extrinsic_matrices(thread_camera_1=self.thread_camera_1, thread_camera_2=self.thread_camera_2)
            if is_ok:
                self.set_extrinsic_matrices(Q1=Q1, Q2=Q2)
            else:
                self.set_extrinsic_matrices()
                tk.messagebox.showinfo(title="Calculo de matrices extrinsecas", message="Posibles problemas:\n\n - El marcador aruco debe estar presente en ambas camaras\n - Los datos para el marcador aruco no son correctos")
        else:
            self.set_extrinsic_matrices()
            tk.messagebox.showinfo(title="Calculo de matrices extrinsecas", message="Ambas camaras deben estar calibradas.")

    # Para eliminar las matrices extrinsecas de las camaras.
    def delete_extrinsic_matrices(self):
        self.set_extrinsic_matrices()
