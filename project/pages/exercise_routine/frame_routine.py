import numpy as np
import customtkinter  as ctk
import os

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from pages.exercise_routine.frame_points_slider_graphic_3D import FramePointsSliderGraphic3D
from pages.exercise_routine.adjust_trajectory import AdjustTrajectory

# ALGORITMOS
from pinhole_camera_model.algorithms.algorithm_mediapipe_pose import AlgorithmMediaPipePose

"""
Frame para mostrar la informacion de la rutina, la clasificacion de la postura para el ejercicio y la trayectoria ajustada del movimiento a realizar
"""
class FrameRoutine(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.data={}
        self.routine_dict={}
        self.forward_adjusted_trajectory=True
        
        # Numero de puntos que se obtendran a lo largo de la trayectoria ajustada
        n=15
        nonlinear_models_parameters_file_name=f"nonlinear_models_parameters.npy"
        trajectory_datasets_path=os.path.join(self.app.folder_name_data, *["trajectory_datasets"])
        folders=os.listdir(trajectory_datasets_path)
        self.adjusted_posture_3D_over_time_list_dict={}
        for folder in folders:
            full_path=os.path.join(trajectory_datasets_path, *[folder])
            files=os.listdir(full_path)
            if nonlinear_models_parameters_file_name in files:
                # Se cargan los parametros de los modelos no lineales de las trayectorias ajustadas
                nonlinear_models_parameters=np.load(os.path.join(full_path, *[nonlinear_models_parameters_file_name]))
                files.remove(nonlinear_models_parameters_file_name)
                files.sort()
                # Se obtienen los puntos 3D de cada archivo
                posture_3D_over_time_list=[np.load(os.path.join(full_path, *[files[i]])) for i in range(len(files))]
                # Se ajusta la trayectoria de cada punto 3D
                adjusted_posture_3D_over_time_list,_=AdjustTrajectory.get_adjusted_posture_3D_and_point_3D_over_time_list(posture_3D_over_time_list, nonlinear_models_parameters, AlgorithmMediaPipePose.number_points, n=n)
                self.adjusted_posture_3D_over_time_list_dict[folder]=adjusted_posture_3D_over_time_list
            else:
               self.adjusted_posture_3D_over_time_list_dict[folder]=None

        self.textbox_info_routine=ctk.CTkTextbox(master=self, width=350, height=200, wrap='word', state="disabled")

        frame_container_colors=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,3), arr=None))
        # Indica en verde si ya ha realizado la postura neutral inicial
        self.label_1=ctk.CTkLabel(master=frame_container_colors,  text="", width=50, height=50, fg_color="red")
        # Indica en verde si ya ha realizado la postura de movimiento
        self.label_2=ctk.CTkLabel(master=frame_container_colors,  text="", width=50, height=50, fg_color="red")
        # Indica en verde si ya ha realizado la postura neutral final
        self.label_3=ctk.CTkLabel(master=frame_container_colors,  text="", width=50, height=50, fg_color="red")
        frame_container_colors.insert_element(cad_pos="0,0", element=self.label_1, padx=5, pady=5, sticky="")
        frame_container_colors.insert_element(cad_pos="0,1", element=self.label_2, padx=5, pady=5, sticky="")
        frame_container_colors.insert_element(cad_pos="0,2", element=self.label_3, padx=5, pady=5, sticky="")

        frame_trajectory=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        label_trajectory=ctk.CTkLabel(master=frame_trajectory, text="Trayectoria ajustada", font=ctk.CTkFont(size=20, weight='bold'))
        self.frame_points_slider_graphic_3D_adjusted_trajectory=FramePointsSliderGraphic3D(master=frame_trajectory, name="FramePointsSliderGraphic3D-AdjustedTrajectory-Classification", square_size=1, width=300, height=250, rate_ms=500)
        frame_trajectory.insert_element(cad_pos="0,0", element=label_trajectory, padx=5, pady=5, sticky="")
        frame_trajectory.insert_element(cad_pos="1,0", element=self.frame_points_slider_graphic_3D_adjusted_trajectory, padx=5, pady=5, sticky="")
        
        self.insert_element(cad_pos="0,0", element=self.textbox_info_routine, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=frame_container_colors, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=frame_trajectory, padx=5, pady=5, sticky="")

    # Para establecer una rutina
    def set_routine(self, data):
        # Se guardan los datos de la rutina
        self.data=data
        m=self.data["exercises_routine"].shape[0]
        self.routine_dict={}
        for i in range(m):
            p1,p2,n=self.data["exercises_routine"][i]
            self.routine_dict[f"{str(p1)}_{str(p2)}"]=[-1 for j in range(n)]

        # Inicializacion
        self.set_text()
        self.init_colors()
        self.forward_adjusted_trajectory=True
        self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(None)
        
    # Para establecer el texto que ira en el cuadro de texto
    def set_text(self):
        self.textbox_info_routine.configure(state="normal")
        self.textbox_info_routine.delete("0.0", "end")

        # Datos de la rutina
        cad=f"NOMBRE:\n{self.data['name_routine']}\n\nDESCRIPCION:\n{self.data['description_routine']}\nRUTINA:"
        m=self.data["exercises_routine"].shape[0]
        for i in range(m):
            p1,p2,n=self.data["exercises_routine"][i]
            cad+=f"\n\n{p1} ➔ {p2}: "
            for j in range(n):
                # Se colcoa "_" si no ha realizado el ejercicio, y se coloca "✔" si ya ha realizado el ejercicio
                cad+="_  " if self.routine_dict[f"{str(p1)}_{str(p2)}"][j] == -1 else "✔  "
        
        self.textbox_info_routine.insert("0.0", cad)
        self.textbox_info_routine.configure(state="disabled")

    # Para establecer los colores a rojo (inicializacion de colores)
    def init_colors(self):
        self.label_1.configure(fg_color="red")
        self.label_2.configure(fg_color="red")
        self.label_3.configure(fg_color="red")

    # Para procesar las predicciones de postura actuales
    def process_predictions(self, predictions_dict):
        # Umbral
        threshold=0.8
        for key in self.routine_dict.keys():
            p1,p2=list(map(lambda elem: int(elem), key.split("_")))
            n=len(self.routine_dict[key])
            for j in range(n):
                if self.routine_dict[key][j] == -1:
                    # Se solicita que haga la postura neutral inicial
                    if self.label_1.cget("fg_color") == "red":
                        pred=predictions_dict[f"neutral_posture_{str(p1).zfill(4)}"]
                        # Si la prediccion para la clase positiva es mayor o igual a 'threshold' (umbral) se cuenta como postura valida
                        if pred[1] >= threshold:
                            self.label_1.configure(fg_color="green")
                    # Se solicita que haga la postura de movimiento o ejecucion
                    elif self.label_2.cget("fg_color") == "red":
                        if self.forward_adjusted_trajectory:
                            adjusted_posture_3D_over_time_list=self.adjusted_posture_3D_over_time_list_dict[f"neutral_posture_{str(p1).zfill(4)}__movement_posture_{str(p2).zfill(4)}"]
                            # Se pasan los puntos 3D ajustados para graficarlos 
                            self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(points_3D_list=adjusted_posture_3D_over_time_list, play_value=1)
                            self.forward_adjusted_trajectory=False
                        pred=predictions_dict[f"movement_posture_{str(p2).zfill(4)}"]
                        # Si la prediccion para la clase positiva es mayor o igual a 'threshold' (umbral) se cuenta como postura valida
                        if pred[1] >= threshold:
                            self.label_2.configure(fg_color="green")
                    # Se solicita que haga la postura neutral final
                    elif self.label_3.cget("fg_color") == "red":
                        if not self.forward_adjusted_trajectory:
                            self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(None)
                            adjusted_posture_3D_over_time_list=self.adjusted_posture_3D_over_time_list_dict[f"neutral_posture_{str(p1).zfill(4)}__movement_posture_{str(p2).zfill(4)}"]
                            # Lista en sentido contrario
                            adjusted_posture_3D_over_time_list=adjusted_posture_3D_over_time_list[::-1]
                            # Se pasan los puntos 3D ajustados para graficarlos 
                            self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(points_3D_list=adjusted_posture_3D_over_time_list, play_value=1)
                            self.forward_adjusted_trajectory=True
                        pred=predictions_dict[f"neutral_posture_{str(p1).zfill(4)}"]
                        # Si la prediccion para la clase positiva es mayor o igual a 'threshold' (umbral) se cuenta como postura valida
                        if pred[1] >= threshold:
                            self.label_3.configure(fg_color="green")
                    else:
                        # Ejercicio completado (reiniciamos variables)
                        self.routine_dict[key][j]=1
                        self.set_text()
                        self.init_colors()
                        self.forward_adjusted_trajectory=True
                        self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(None)
                    return