import numpy as np
import customtkinter  as ctk
import os
import sys
import pickle

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from pages.exercise_routine.frame_routine import FrameRoutine
from pages.exercise_routine.frame_show_posture_classification import FrameShowPostureClassification
# Modelo de red neuronal
from framework_ml.algorithms.NN import Sequential

"""
Frame para mostrar la clasificacion de la postura actual
"""
class FrameExerciseClassification(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        
        # Directorios importantes
        framework_ml_path=os.path.join(self.app.current_path, *["framework_ml"])
        models_path=os.path.join(self.app.folder_name_data, *["models"])
        routines_path=os.path.join(self.app.folder_name_data, *["routines"])

        # Agregar ruta para la libreria de ML
        if framework_ml_path not in sys.path:
            sys.path.insert(0, framework_ml_path)

        # Carga de modelos de clasificacion
        files=os.listdir(models_path)
        self.models_dict={}
        for file in files:
            model_name=file.split(".")[0]
            self.models_dict[model_name]=Sequential.load(path=models_path, file_name=model_name)

        # Carga de rutinas
        files=os.listdir(routines_path)
        self.routines_dict={}
        for file in files:
            routine_name=file.split(".")[0]
            full_path=os.path.join(routines_path, *[f"{routine_name}.pkl"])
            if os.path.exists(full_path):
                with open(full_path, "rb") as f:
                    # {"exercises_routine": array, "name_routine": "...", "description_routine": "..."}
                    data=pickle.load(f) 
                    self.routines_dict[routine_name]=data
            else:
                self.routines_dict[routine_name]=None

        # Frame para mostrar la rutina seleccionada, la clasificacion de cada postura para cada ejercicio y la trayectoria ajustada del movimiento a ejecutar
        frame_container_routines=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,1), arr=None))
        label_routines=ctk.CTkLabel(master=frame_container_routines,  text="Rutinas", font=ctk.CTkFont(size=25, weight='bold'))
        self.var_routine=ctk.StringVar(value="")
        self.var_routine.trace_add("write", self.trace_var_routine)
        option_menu_routines=ctk.CTkOptionMenu(master=frame_container_routines, values=sorted(list(self.routines_dict.keys())), variable=self.var_routine, command=None)
        self.frame_routine=FrameRoutine(master=frame_container_routines, name="FrameRoutine")
        frame_container_routines.insert_element(cad_pos="0,0", element=label_routines, padx=5, pady=5, sticky="")
        frame_container_routines.insert_element(cad_pos="1,0", element=option_menu_routines, padx=5, pady=5, sticky="")
        frame_container_routines.insert_element(cad_pos="2,0", element=self.frame_routine, padx=5, pady=5, sticky="")

        # Frame para mostrar las predicciones de la postura neutra
        frame_container_neutral_posture=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,1), arr=None))
        label_neutral_posture=ctk.CTkLabel(master=frame_container_neutral_posture,  text="Postura neutral", font=ctk.CTkFont(size=20, weight='bold'))
        self.frame_show_posture_classification_neutral_posture_1=FrameShowPostureClassification(master=frame_container_neutral_posture, name="FrameShowPostureClassificationNeutralPosture1", title="Postura neutral 1")
        frame_container_neutral_posture.insert_element(cad_pos="0,0", element=label_neutral_posture, padx=5, pady=5, sticky="")
        frame_container_neutral_posture.insert_element(cad_pos="1,0", element=self.frame_show_posture_classification_neutral_posture_1, padx=5, pady=5, sticky="")

        # Frame para mostrar las predicciones de las posturas de movimiento 
        frame_container_movement_posture=CreateFrame(master=self, grid_frame=GridFrame(dim=(3,2), arr=np.array([["0,0","0,0"],["1,0","1,1"],["2,0","2,0"]])))
        label_movement_posture=ctk.CTkLabel(master=frame_container_movement_posture,  text="Postura de movimiento", font=ctk.CTkFont(size=20, weight='bold'))
        self.frame_show_posture_classification_movement_posture_1=FrameShowPostureClassification(master=frame_container_movement_posture, name="FrameShowPostureClassificationMovementPosture1", title="Postura mov 1")
        self.frame_show_posture_classification_movement_posture_2=FrameShowPostureClassification(master=frame_container_movement_posture, name="FrameShowPostureClassificationMovementPosture2", title="Postura mov 2")
        self.frame_show_posture_classification_movement_posture_3=FrameShowPostureClassification(master=frame_container_movement_posture, name="FrameShowPostureClassificationMovementPosture3", title="Postura mov 3")
        frame_container_movement_posture.insert_element(cad_pos="0,0", element=label_movement_posture, padx=5, pady=5, sticky="")
        frame_container_movement_posture.insert_element(cad_pos="1,0", element=self.frame_show_posture_classification_movement_posture_1, padx=5, pady=5, sticky="")
        frame_container_movement_posture.insert_element(cad_pos="1,1", element=self.frame_show_posture_classification_movement_posture_2, padx=5, pady=5, sticky="")
        frame_container_movement_posture.insert_element(cad_pos="2,0", element=self.frame_show_posture_classification_movement_posture_3, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=frame_container_routines, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=frame_container_neutral_posture, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=frame_container_movement_posture, padx=5, pady=5, sticky="")
    
    # Para detectar cuando se selecciona una rutina
    def trace_var_routine(self, var, index, mode):
        data=self.routines_dict[self.var_routine.get()]
        self.frame_routine.set_routine(data)

    # Para procesar los datos de los algoritmos
    def process_data(self, algorithm):
        if algorithm is not None:
            if algorithm.points_3D is not None:
                predictions_dict={}
                # Ejemplo nuevo (1,42)
                X=algorithm.classification_euler_angles.flatten(order='C')[None,:]
                # Postura neutral
                for i,frame_show_posture_classification_neutral_posture in enumerate([
                        self.frame_show_posture_classification_neutral_posture_1
                    ]):
                    # Nombre de la postura
                    posture_name=f"neutral_posture_{str(i + 1).zfill(4)}"
                    # Predicion para el ejemplo nuevo (para la clase 0 y para la clase 1)
                    pred=self.models_dict[posture_name].predict_proba(X).flatten()
                    frame_show_posture_classification_neutral_posture.update_prediction(pred)
                    predictions_dict[posture_name]=pred
                # Postura de movimiento
                for i,frame_show_posture_classification_movement_posture in enumerate([
                        self.frame_show_posture_classification_movement_posture_1,
                        self.frame_show_posture_classification_movement_posture_2,
                        self.frame_show_posture_classification_movement_posture_3
                    ]):
                    # Nombre de la postura
                    posture_name=f"movement_posture_{str(i + 1).zfill(4)}"
                    # Predicion para el ejemplo nuevo (para la clase 0 y para la clase 1)
                    pred=self.models_dict[posture_name].predict_proba(X).flatten()
                    frame_show_posture_classification_movement_posture.update_prediction(pred)
                    predictions_dict[posture_name]=pred
                # Procesamos predicciones
                self.frame_routine.process_predictions(predictions_dict)