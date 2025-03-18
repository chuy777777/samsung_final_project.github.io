import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.frame_graphic_3D import FrameGraphic3D
from pages.estimation.frame_select_angles import FrameSelectAngles

# ALGORITMOS
from pinhole_camera_model.algorithms.algorithm_mediapipe_pose import AlgorithmMediaPipePose

"""
Frame para mostrar las estimaciones 3D con diferentes opciones para configurar.
"""
class FrameEstimationGraphic3DWithOptions(CreateFrame):
    def __init__(self, master, name, callback=None, show_select_angles=True, square_size=1, width=400, height=400, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        self.callback=callback
        self.show_select_angles=show_select_angles

        self.var_points=ctk.IntVar(value=1)
        self.var_points.trace_add("write", self.trace_vars)
        self.var_lines=ctk.IntVar(value=1)
        self.var_lines.trace_add("write", self.trace_vars)
        self.var_coordinate_systems=ctk.IntVar(value=0)
        self.var_coordinate_systems.trace_add("write", self.trace_vars)
        frame_options=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,2), arr=None))
        check_box_points=ctk.CTkCheckBox(master=frame_options, text="Puntos", variable=self.var_points, onvalue=1, offvalue=0)
        check_box_lines=ctk.CTkCheckBox(master=frame_options, text="Lineas", variable=self.var_lines, onvalue=1, offvalue=0)
        check_box_coordinate_systems=ctk.CTkCheckBox(master=frame_options, text="Sistemas de coordenadas", variable=self.var_coordinate_systems, onvalue=1, offvalue=0)
        frame_options.insert_element(cad_pos="0,0", element=check_box_points, padx=5, pady=5, sticky="w")
        frame_options.insert_element(cad_pos="1,0", element=check_box_lines, padx=5, pady=5, sticky="w")
        frame_options.insert_element(cad_pos="0,1", element=check_box_coordinate_systems, padx=5, pady=5, sticky="w")

        self.frame_graphic_3D=FrameGraphic3D(master=self, name="FrameGraphic3DWithOptions", square_size=square_size, width=width, height=height)

        self.insert_element(cad_pos="0,0", element=self.frame_graphic_3D, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=frame_options, padx=5, pady=5, sticky="ew")
        # En caso de especificarse para mostrar angulos articulares
        if self.show_select_angles:
            self.var_angles=ctk.IntVar(value=0)
            self.var_angles.trace_add("write", self.trace_vars)
            check_box_angles=ctk.CTkCheckBox(master=frame_options, text="Angulos", variable=self.var_angles, onvalue=1, offvalue=0)
            frame_options.insert_element(cad_pos="1,1", element=check_box_angles, padx=5, pady=5, sticky="w")
            self.frame_select_angles=FrameSelectAngles(master=self, name="FrameSelectAngles", algorithm_name=AlgorithmMediaPipePose.algorithm_name, number_points=AlgorithmMediaPipePose.number_points, callback=None)
            self.insert_element(cad_pos="2,0", element=self.frame_select_angles, padx=5, pady=5, sticky="ew")

    # Para notificar por medio de un 'callback' cuando las opciones cambien.
    def trace_vars(self, var, index, mode):
        if self.callback is not None:
            self.callback()

    # Para dibujar informacion relevante 
    def draw_estimation(self, algorithm):
        self.frame_graphic_3D.clear()
        if algorithm is not None:
            if algorithm.points_3D is not None:
                # Dibujar puntos 
                if self.var_points.get():
                    self.frame_graphic_3D.plot_points(ps=algorithm.points_3D, color_rgb=(255,0,0), alpha=0.8, marker=".", s=70)
                # Dibujar lineas 
                if self.var_lines.get():
                    self.frame_graphic_3D.plot_lines(ps=algorithm.points_3D, connection_list=algorithm.connection_list, color_rgb=(0,255,0))
                # Dibujar sistemas de coordenadas
                if self.var_coordinate_systems.get():
                    for i in range(len(algorithm.coordinate_system_list)):
                        length=0.1
                        self.frame_graphic_3D.plot_coordinate_system(t_list=[algorithm.points_3D[[i],:].T], T_list=[algorithm.coordinate_system_list[i]], length=length)
                # Dibujar angulos
                # En caso de especificarse para mostrar angulos articulares
                if self.show_select_angles:
                    if self.var_angles.get():
                        for triplet in self.frame_select_angles.triplet_list:
                            p1,pm,p2=algorithm.points_3D[triplet[0]],algorithm.points_3D[triplet[1]],algorithm.points_3D[triplet[2]]
                            self.frame_graphic_3D.plot_angle(pm=pm, p1=p1, p2=p2, show_degree=True, color_rgb_polygon_facecolors=(0,255,0), color_rgb_polygon_edgecolors=(0,0,0), alpha_polygon=0.2, color_rgb_text=(0,0,0), fontsize_text="medium", fontweight_text="bold")
        self.frame_graphic_3D.draw()