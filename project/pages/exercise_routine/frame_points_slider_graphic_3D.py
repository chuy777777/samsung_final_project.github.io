import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.frame_graphic_3D import FrameGraphic3D

# ALGORITMOS
from pinhole_camera_model.algorithms.algorithm_mediapipe_pose import AlgorithmMediaPipePose

"""
Frame para graficar posturas 3D 
"""
class FramePointsSliderGraphic3D(CreateFrame):
    def __init__(self, master, name, square_size=1, width=400, height=400, rate_ms=500, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(4,1), arr=None), **kwargs)
        self.width=width
        self.height=height
        self.points_3D_list=None
        self.trajectory_parameters=None
        self.slider_frame=None
        self.in_loop=False
        self.rate_ms=rate_ms

        self.frame_graphic_3D=FrameGraphic3D(master=self, name=f"FrameGraphic3D-{name}", square_size=square_size, width=width, height=height)
        self.var_play=ctk.IntVar(value=0)
        self.var_play.trace_add("write", self.trace_var)
        self.check_box_play=ctk.CTkCheckBox(master=self, text="Movimiento automatico", variable=self.var_play, onvalue=1, offvalue=0, state="disabled")
        self.var_total_data=ctk.StringVar(value="")
        label_total_data=ctk.CTkLabel(master=self, textvariable=self.var_total_data, font=ctk.CTkFont(size=16, weight='bold'))

        self.insert_element(cad_pos="0,0", element=self.frame_graphic_3D, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=self.check_box_play, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=label_total_data, padx=5, pady=5, sticky="")

    # Para detectar cuando se quiera recorrer cada postura de manera automatica
    def trace_var(self, var, index, mode):
        if self.var_play.get(): 
            self.in_loop=True
            self.loop()
        else:
            self.in_loop=False

    # Para mostrar cada postura cada cierto tiempo
    def loop(self):
        if self.points_3D_list is not None:
            current_value=int(self.slider_frame.get())
            if current_value + 1 <= len(self.points_3D_list) - 1:
                value=current_value + 1
            else:
                value=0
            self.slider_callback(value=value)
            if self.in_loop:
                self.after(self.rate_ms, self.loop)

    # Para establecer los datos de las posturas
    def set_data(self, points_3D_list, play_value=0):
        self.var_play.set(value=play_value)
        self.points_3D_list=points_3D_list
        self.destroy_element(cad_pos="3,0")
        if points_3D_list is not None:
            self.var_total_data.set(value=f"Total: {len(points_3D_list)}")
            self.check_box_play.configure(state="normal")
            self.slider_frame=ctk.CTkSlider(master=self, from_=0, to=len(points_3D_list) - 1, width=self.width, number_of_steps=len(points_3D_list) - 1, command=self.slider_callback)
            self.insert_element(cad_pos="3,0", element=self.slider_frame, padx=5, pady=5, sticky="")
            self.slider_callback(value=0)
        else:
            self.var_total_data.set(value="")
            self.check_box_play.configure(state="disabled")
            self.frame_graphic_3D.clear()
            self.frame_graphic_3D.draw()
        if play_value:
            self.trace_var(None,None,None)

    # Para detectar cuando se mueve el 'slice' 
    def slider_callback(self, value):
        if len(self.points_3D_list) > 0 and int(value) < len(self.points_3D_list):
            self.slider_frame.set(value)
            points_3D=self.points_3D_list[int(value)]
            self.frame_graphic_3D.clear()
            # Dibujar puntos 
            self.frame_graphic_3D.plot_points(ps=points_3D, color_rgb=(255,0,0), alpha=0.8, marker=".", s=70)
            # Dibujar lineas 
            self.frame_graphic_3D.plot_lines(ps=points_3D, connection_list=AlgorithmMediaPipePose.connection_list, color_rgb=(0,255,0))
            self.frame_graphic_3D.draw()