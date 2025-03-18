import customtkinter  as ctk
import numpy as np
import os
import pickle

from components.create_frame import CreateFrame
from components.create_scrollable_frame import CreateScrollableFrame
from components.grid_frame import GridFrame
from pages.exercise_routine.adjust_trajectory import AdjustTrajectory
from pages.exercise_routine.frame_points_slider_graphic_3D import FramePointsSliderGraphic3D

# ALGORITMOS
from pinhole_camera_model.algorithms.algorithm_mediapipe_pose import AlgorithmMediaPipePose

"""
Frame para mostrar graficas de todos los conjuntos de datos recabados y para crear rutinas
"""
class FrameExerciseRoutine(CreateScrollableFrame):
    def __init__(self, master, name, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(4,2), arr=np.array([["0,0","0,0"],["1,0","1,1"],["2,0","2,1"],["3,0","3,0"]])), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.rate_ms=50

        # Directorios importantes
        self.folder_names_datasets=os.listdir(os.path.join(self.app.folder_name_data, *["datasets"]))
        self.folder_names_neutral_posture=list(filter(lambda elem: elem.split('_')[0] == 'neutral', self.folder_names_datasets))
        self.folder_names_movement_posture=list(filter(lambda elem: elem.split('_')[0] == 'movement', self.folder_names_datasets))
        self.folder_names_neutral_posture.sort()
        self.folder_names_movement_posture.sort()

        # Frame para graficar todas las posturas aleatorias
        frame_all_postures=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        label_title=ctk.CTkLabel(master=frame_all_postures, text="Posturas aleatorias", font=ctk.CTkFont(size=25, weight='bold'))
        self.frame_points_slider_graphic_3D_all_postures=FramePointsSliderGraphic3D(master=frame_all_postures, name="FramePointsSliderGraphic3D-AllPostures", square_size=1, width=400, height=300)
        frame_all_postures.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        frame_all_postures.insert_element(cad_pos="1,0", element=self.frame_points_slider_graphic_3D_all_postures, padx=5, pady=5, sticky="")
        full_path=os.path.join(self.app.folder_name_data, *["datasets", "all_postures"])
        files=list(filter(lambda elem: elem.split("_")[0] != "euler", os.listdir(full_path)))
        files.sort()
        points_3D_list=[np.load(os.path.join(full_path, *[files[i]])) for i in range(len(files))]
        self.frame_points_slider_graphic_3D_all_postures.set_data(points_3D_list=points_3D_list)

        # Frame para graficar todas las posturas de una postura neutra especifica
        frame_neutral_posture=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,1), arr=None))
        label_title=ctk.CTkLabel(master=frame_neutral_posture, text="Posturas neutrales", font=ctk.CTkFont(size=25, weight='bold'))
        self.values_neutral_posture=list(map(lambda elem: f'Postura {int(elem.split("_")[2])}', self.folder_names_neutral_posture))
        self.var_neutral_posture=ctk.StringVar(value="")
        option_menu_neutral_posture=ctk.CTkOptionMenu(master=frame_neutral_posture, values=self.values_neutral_posture, variable=self.var_neutral_posture, command=None)
        self.frame_points_slider_graphic_3D_neutral_posture=FramePointsSliderGraphic3D(master=frame_neutral_posture, name="FramePointsSliderGraphic3D-NeutralPosture", square_size=1, width=400, height=300)
        self.textbox_info_neutral_posture=ctk.CTkTextbox(master=frame_neutral_posture, width=350, height=100, wrap='word')
        self.var_neutral_posture.trace_add("write", lambda var, index, mode: self.posture_changed(self.var_neutral_posture, self.folder_names_neutral_posture, self.textbox_info_neutral_posture, self.frame_points_slider_graphic_3D_neutral_posture))
        frame_neutral_posture.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        frame_neutral_posture.insert_element(cad_pos="1,0", element=option_menu_neutral_posture, padx=5, pady=5, sticky="")
        frame_neutral_posture.insert_element(cad_pos="2,0", element=self.frame_points_slider_graphic_3D_neutral_posture, padx=5, pady=5, sticky="")
        frame_neutral_posture.insert_element(cad_pos="3,0", element=self.textbox_info_neutral_posture, padx=5, pady=5, sticky="")

        # Frame para graficar todas las posturas de una postura de movimiento especifica
        frame_movement_posture=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,1), arr=None))
        label_title=ctk.CTkLabel(master=frame_movement_posture, text="Posturas de movimiento", font=ctk.CTkFont(size=25, weight='bold'))
        self.values_movement_posture=list(map(lambda elem: f'Postura {int(elem.split("_")[2])}', self.folder_names_movement_posture))
        self.var_movement_posture=ctk.StringVar(value="")
        option_menu_movement_posture=ctk.CTkOptionMenu(master=frame_movement_posture, values=self.values_movement_posture, variable=self.var_movement_posture, command=None)
        self.frame_points_slider_graphic_3D_movement_posture=FramePointsSliderGraphic3D(master=frame_movement_posture, name="FramePointsSliderGraphic3D-NeutralPosture", square_size=1, width=400, height=300)
        self.textbox_info_movement_posture=ctk.CTkTextbox(master=frame_movement_posture, width=350, height=100, wrap='word')
        self.var_movement_posture.trace_add("write", lambda var, index, mode: self.posture_changed(self.var_movement_posture, self.folder_names_movement_posture, self.textbox_info_movement_posture, self.frame_points_slider_graphic_3D_movement_posture))
        frame_movement_posture.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        frame_movement_posture.insert_element(cad_pos="1,0", element=option_menu_movement_posture, padx=5, pady=5, sticky="")
        frame_movement_posture.insert_element(cad_pos="2,0", element=self.frame_points_slider_graphic_3D_movement_posture, padx=5, pady=5, sticky="")
        frame_movement_posture.insert_element(cad_pos="3,0", element=self.textbox_info_movement_posture, padx=5, pady=5, sticky="")

        # Frame para graficar todas las posturas de una cierta trayectoria
        frame_trajectory=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        label_title=ctk.CTkLabel(master=frame_trajectory, text="Trayectoria", font=ctk.CTkFont(size=25, weight='bold'))
        self.frame_points_slider_graphic_3D_trajectory=FramePointsSliderGraphic3D(master=frame_trajectory, name="FramePointsSliderGraphic3D-Trajectory", square_size=1, width=400, height=300)
        frame_trajectory.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        frame_trajectory.insert_element(cad_pos="1,0", element=self.frame_points_slider_graphic_3D_trajectory, padx=5, pady=5, sticky="")

        # Frame para graficar todas las posturas de una cierta trayectoria ajustada
        frame_adjusted_trajectory=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        label_title=ctk.CTkLabel(master=frame_adjusted_trajectory, text="Trayectoria ajustada", font=ctk.CTkFont(size=25, weight='bold'))
        self.frame_points_slider_graphic_3D_adjusted_trajectory=FramePointsSliderGraphic3D(master=frame_adjusted_trajectory, name="FramePointsSliderGraphic3D-AdjustedTrajectory", square_size=1, width=400, height=300)
        frame_adjusted_trajectory.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        frame_adjusted_trajectory.insert_element(cad_pos="1,0", element=self.frame_points_slider_graphic_3D_adjusted_trajectory, padx=5, pady=5, sticky="")

        # Frame para mostrar, agregar y eliminar rutinas
        frame_routine=CreateFrame(master=self, grid_frame=GridFrame(dim=(7,2), arr=np.array([["0,0","0,0"],["1,0","1,0"],["2,0","2,0"],["3,0","3,1"],["4,0","4,1"],["5,0","5,1"],["6,0","6,1"]])))
        button_create_routine=ctk.CTkButton(master=frame_routine, text="Agregar", fg_color="green2", hover_color="green3", command=self.create_routine)
        label_create_routine=ctk.CTkLabel(master=frame_routine, text="# Postura neutra,# Postura de movimiento,# Repeticiones (tripletas separadas por un salto de linea)", font=ctk.CTkFont(size=12, weight='bold'))
        self.textbox_create_routine=ctk.CTkTextbox(master=frame_routine, width=150, height=100, wrap='word')
        label_name=ctk.CTkLabel(master=frame_routine, text="Nombre de referencia", font=ctk.CTkFont(size=12, weight='bold'))
        self.var_name_routine=ctk.StringVar(value="")
        entry_name_routine=ctk.CTkEntry(master=frame_routine, width=350, textvariable=self.var_name_routine)
        label_description_routine=ctk.CTkLabel(master=frame_routine, text="Descripcion de referencia", font=ctk.CTkFont(size=12, weight='bold'))
        self.textbox_description_routine=ctk.CTkTextbox(master=frame_routine, width=350, height=100, wrap='word')
        self.var_routine=ctk.StringVar(value="")
        self.var_routine.trace_add("write", self.trace_var_routine)
        self.option_menu_routine=ctk.CTkOptionMenu(master=frame_routine, values=self.get_all_name_routines(), variable=self.var_routine, command=None)
        button_delete_routine=ctk.CTkButton(master=frame_routine, text="Eliminar", fg_color="red2", hover_color="red3", command=self.delete_routine)
        self.textbox_info_routine=ctk.CTkTextbox(master=frame_routine, width=350, height=200, wrap='word', state="disabled")
        self.textbox_exercises_routine=ctk.CTkTextbox(master=frame_routine, width=350, height=200, wrap='word', state="disabled")
        frame_routine.insert_element(cad_pos="0,0", element=button_create_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="1,0", element=label_create_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="2,0", element=self.textbox_create_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="3,0", element=label_name, padx=5, pady=1, sticky="")
        frame_routine.insert_element(cad_pos="4,0", element=entry_name_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="3,1", element=label_description_routine, padx=5, pady=1, sticky="")
        frame_routine.insert_element(cad_pos="4,1", element=self.textbox_description_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="5,0", element=self.option_menu_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="5,1", element=button_delete_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="6,0", element=self.textbox_info_routine, padx=5, pady=5, sticky="")
        frame_routine.insert_element(cad_pos="6,1", element=self.textbox_exercises_routine, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=frame_all_postures, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=frame_neutral_posture, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=frame_movement_posture, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=frame_trajectory, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,1", element=frame_adjusted_trajectory, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=frame_routine, padx=5, pady=5, sticky="")
    
    # Para detectar cuanddo se haya seleccionado una rutina y mostrar dicha informacion de la rutina seleccionada
    def trace_var_routine(self, var, index, mode):
        self.textbox_info_routine.configure(state="normal")
        self.textbox_exercises_routine.configure(state="normal")
        self.textbox_info_routine.delete("0.0", "end")
        self.textbox_exercises_routine.delete("0.0", "end")
        if self.var_routine.get() == "":
            self.option_menu_routine.configure(values=self.get_all_name_routines())
        else:
            index_routine=int(self.var_routine.get().split(" ")[1])
            path=os.path.join(self.app.folder_name_data, *["routines", f"routine_{str(index_routine).zfill(4)}.pkl"])
            with open(path, "rb") as f:
                data=pickle.load(f)
                exercises_routine=data['exercises_routine']
                cad=""
                for i in range(exercises_routine.shape[0]):
                    cad+=f"EJERCICIO {i + 1}\nPostura neutra: {exercises_routine[i][0]}\nPostura de movimiento: {exercises_routine[i][1]}\nNumero de repeticiones: {exercises_routine[i][2]}\n\n"
                self.textbox_info_routine.insert("0.0", f"NOMBRE:\n{data['name_routine']}\n\nDESCRIPCION:\n{data['description_routine']}")
                self.textbox_exercises_routine.insert("0.0", f"{cad}")
        self.textbox_info_routine.configure(state="disabled")
        self.textbox_exercises_routine.configure(state="disabled")

    # Para obtener todos los nombres de las rutinas creadas
    def get_all_name_routines(self):
        files=os.listdir(os.path.join(self.app.folder_name_data, *["routines"]))
        all_name_routines=list(map(lambda elem: f"Rutina {int(elem.split('.')[0].split('_')[1])}", files))
        all_name_routines.sort()
        return all_name_routines

    # Para crear los datos de la rutina que se va a guardar
    def get_data_to_save(self, text_exercises_routine, name_routine, description_routine):
        triplets=[]
        for row in text_exercises_routine.split("\n"):
            split_row=row.split(",")
            if len(split_row) == 3:
                vals=[]
                for str_val in split_row:
                    try:
                        val=int(str_val)
                        vals.append(val)
                    except ValueError:
                        break 
                if len(vals) == 3:
                    val1,val2,_=vals 
                    if val1 in list(map(lambda elem: int(elem.split(" ")[1]), self.values_neutral_posture)) and val2 in list(map(lambda elem: int(elem.split(" ")[1]), self.values_movement_posture)):
                        triplets.append(vals)
        if len(triplets) == 0:
            return None
        else: 
            exercises_routine=np.zeros((len(triplets),3), dtype=int)
            for i,vals in enumerate(triplets):
                exercises_routine[i]=vals
            data={
                "exercises_routine": exercises_routine,
                "name_routine": name_routine,
                "description_routine": description_routine
            }
            return data

    # Para guardar la rutina
    def create_routine(self):
        text_exercises_routine=self.textbox_create_routine.get("0.0", "end")
        name_routine=self.var_name_routine.get()
        description_routine=self.textbox_description_routine.get("0.0", "end")
        data=self.get_data_to_save(text_exercises_routine, name_routine, description_routine)
        if data is not None:
            path=os.path.join(self.app.folder_name_data, *["routines"])
            # Si el directorio no existe entonces se crea
            if not os.path.exists(path):
                os.makedirs(path)
            index_routine=1
            files=os.listdir(path)
            if len(files) > 0:
                files.sort()
                index_routine=int(files[-1].split(".")[0].split("_")[1]) + 1
            path=os.path.join(path, *[f'routine_{str(index_routine).zfill(4)}.pkl'])
            with open(path, "wb") as f:
                pickle.dump(data, f)
                # Reiniciamos valores de formulario
                self.textbox_create_routine.delete("0.0", "end")
                self.var_name_routine.set(value="")
                self.textbox_description_routine.delete("0.0", "end")
                # Reiniciamos valores de rutinas
                self.var_routine.set(value="")

    # Para eliminar la rutina seleccionada
    def delete_routine(self):
        if self.var_routine.get() != "":
            index_routine=int(self.var_routine.get().split(" ")[1])
            path=os.path.join(self.app.folder_name_data, *["routines", f"routine_{str(index_routine).zfill(4)}.pkl"])
            if os.path.exists(path):
                os.remove(path)
                # Reiniciamos valores de rutinas
                self.var_routine.set(value="")

    # Para detectar cuando una postura neutral o de movimiento ha cambiado 
    def posture_changed(self, var_posture, folder_names_posture, textbox_info_posture, frame_points_slider_graphic_3D_posture):
        # Indice de la postura seleccionada
        index_posture=var_posture.get().split(" ")[1].zfill(4)
        folder_name=list(filter(lambda elem: elem.split("_")[2] == index_posture, folder_names_posture))[0]
        full_path=os.path.join(self.app.folder_name_data, *["datasets", folder_name])
        # Nombres de los archivos de la postura
        files=os.listdir(full_path)
        # Diccionario con la informacion de la postura (nombre y descripcion)
        d=None 
        dict_name=f'{folder_name}.pkl'
        if dict_name in files:
            with open(os.path.join(full_path, *[dict_name]), "rb") as f:
                d=pickle.load(f)
            files.remove(dict_name)
        textbox_info_posture.configure(state="normal")
        textbox_info_posture.delete("0.0", "end")
        if d is not None:
            textbox_info_posture.insert("0.0", f"NOMBRE:\n{d['name']}\n\nDESCRIPCION:\n{d['description']}")
        textbox_info_posture.configure(state="disabled")
        # Se filtra para solo obtener los nombres de los archivos que tienen puntos 3D
        files=list(filter(lambda elem: elem.split("_")[0] != "euler", files))
        # Se ordenan los nombres
        files.sort()
        # Se obtienen los puntos 3D de cada archivo
        points_3D_list=[np.load(os.path.join(full_path, *[files[i]])) for i in range(len(files))]
        # Se pasan los puntos 3D para graficarlos 
        frame_points_slider_graphic_3D_posture.set_data(points_3D_list=points_3D_list)

        # Busqueda de trayectoria
        if self.var_neutral_posture.get() != '' and self.var_movement_posture.get() != '':
            trajectory_full_path=os.path.join(self.app.folder_name_data, *["trajectory_datasets"])
            folders=os.listdir(trajectory_full_path)
            neutral_posture_index=self.var_neutral_posture.get().split(" ")[1].zfill(4)
            movement_posture_index=self.var_movement_posture.get().split(" ")[1].zfill(4)
            # Nombre de la carpeta que cuenta con los puntos 3D de la trayectoria descrita por la postura neutra y la postura de movimiento
            folder_name=f"neutral_posture_{neutral_posture_index}__movement_posture_{movement_posture_index}"
            if folder_name in folders:
                full_path=os.path.join(trajectory_full_path, *[folder_name])
                # Nombres de los archivos de la trayectoria
                files=os.listdir(full_path)
                # Nombre del archivo que cuenta con los parametros del modelo no lineal que describe la trayectoria ajustada
                nonlinear_models_parameters_file_name=f"nonlinear_models_parameters.npy"
                band=False
                if nonlinear_models_parameters_file_name in files:
                    band=True
                    files.remove(nonlinear_models_parameters_file_name)
                files.sort()
                # Se obtienen los puntos 3D de cada archivo
                posture_3D_over_time_list=[np.load(os.path.join(full_path, *[files[i]])) for i in range(len(files))]
                # Se pasan los puntos 3D para graficarlos 
                self.frame_points_slider_graphic_3D_trajectory.set_data(points_3D_list=posture_3D_over_time_list)
                # Se cargan los parametros del modelo no lineal para la trayectoria si es que existen
                if band:
                    nonlinear_models_parameters=np.load(os.path.join(full_path, *[nonlinear_models_parameters_file_name]))
                    # Numero de puntos que se obtendran a lo largo de la trayectoria ajustada
                    n=100
                    adjusted_posture_3D_over_time_list,adjusted_point_3D_over_time_list=AdjustTrajectory.get_adjusted_posture_3D_and_point_3D_over_time_list(posture_3D_over_time_list, nonlinear_models_parameters, AlgorithmMediaPipePose.number_points, n=n)
                    # Se pasan los puntos 3D ajustados para graficarlos 
                    self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(points_3D_list=adjusted_posture_3D_over_time_list)
                else:
                    self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(points_3D_list=None)
            else:
                self.frame_points_slider_graphic_3D_trajectory.set_data(points_3D_list=None)
                self.frame_points_slider_graphic_3D_adjusted_trajectory.set_data(points_3D_list=None)