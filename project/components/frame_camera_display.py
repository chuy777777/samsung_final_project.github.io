import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
import components.utils as utils

"""
Frame para mostrar lo que captura una camara.
"""
class FrameCameraDisplay(CreateFrame):
    def __init__(self, master, name, thread_camera, rate_ms=100, scale_percent=100, editable=False, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        self.thread_camera=thread_camera
        self.rate_ms=rate_ms
        self.scale_percent=scale_percent
        self.editable=editable
        self.frame_bgr=None

        label_camera_view=ctk.CTkLabel(master=self, text="")
        
        self.insert_element(cad_pos="0,0", element=label_camera_view, padx=5, pady=5, sticky="")

        self.show_camera_view()

    # Para actualizar la imagen mostrada
    def update_label_camera(self, frame_bgr):
        if frame_bgr is not None:
            # Para reescalar la imagen en caso de ser necesario
            if self.scale_percent != 100:
                frame_bgr=utils.resize_frame_bgr(scale_percent=self.scale_percent, frame_bgr=frame_bgr)
                self.frame_bgr=frame_bgr
            img=utils.frame_bgr_to_ctk_img(frame_bgr=frame_bgr)
            self.get_element(cad_pos="0,0").configure(image=img)
            self.get_element(cad_pos="0,0").image=img
        else:
            self.frame_bgr=None
            self.get_element(cad_pos="0,0").configure(image=None)
            self.get_element(cad_pos="0,0").image=None

    # Se llama cada cierto tiempo a esta funcion      
    def show_camera_view(self):
        self.frame_bgr=self.thread_camera.frame_bgr 
        # En caso de ser editable nosotros mismos tenemos que llamar al metodo 'update_label_camera'
        if not self.editable:
            self.update_label_camera(frame_bgr=self.frame_bgr)
        self.after(self.rate_ms, self.show_camera_view)  