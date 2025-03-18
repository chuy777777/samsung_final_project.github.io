import customtkinter  as ctk
import numpy as np
from pynput import keyboard
import os
import pickle

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from pages.estimation.thread_keyboard import ThreadKeyboard

"""
Frame para guardar las estimaciones 3D y los angulos de Euler para la clasificacion.

El control remoto que se utiliza tiene dos botones:
    - Para subir el volumen (boton mas grande)
        Este boton es el que se usara para guardar los puntos 3D.
    - Para bajar el volumen (boton mas pequeño)
"""
class FrameSaveEstimations(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.algorithm=[]
        self.thread_keyboard=None
        self.keyboard_controller=keyboard.Controller()

        # Para indicar cuando se desean guardar datos
        self.var_save_estimations=ctk.IntVar(value=0)
        self.var_save_estimations.trace_add("write", self.trace_var)
        check_box_save_estimations=ctk.CTkCheckBox(master=self, text="Guardar estimaciones 3D", variable=self.var_save_estimations, onvalue=1, offvalue=0)

        # Frame con toda la informacion necesaria para guardar datos
        self.frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(12,1), arr=None))
        button_reset=ctk.CTkButton(master=self.frame_container, text="Reiniciar", fg_color="red2", hover_color="red3", command=self.reset_all)
        label_folder_name=ctk.CTkLabel(master=self.frame_container, text="Nombre de la carpeta", font=ctk.CTkFont(size=12, weight='bold'))
        self.var_folder_name=ctk.StringVar(value="")
        entry_folder_name=ctk.CTkEntry(master=self.frame_container, width=350, textvariable=self.var_folder_name)
        label_file_name=ctk.CTkLabel(master=self.frame_container, text="Nombre del archivo", font=ctk.CTkFont(size=12, weight='bold'))
        self.var_file_name=ctk.StringVar(value="")
        entry_file_name=ctk.CTkEntry(master=self.frame_container, width=350, textvariable=self.var_file_name)
        self.var_trajectory=ctk.IntVar(value=0)
        check_box_trajectory=ctk.CTkCheckBox(master=self.frame_container, text="Datos para trayectoria", variable=self.var_trajectory, onvalue=1, offvalue=0)
        label_name=ctk.CTkLabel(master=self.frame_container, text="Nombre de referencia", font=ctk.CTkFont(size=12, weight='bold'))
        self.var_name=ctk.StringVar(value="")
        entry_name=ctk.CTkEntry(master=self.frame_container, width=350, textvariable=self.var_name)
        label_description=ctk.CTkLabel(master=self.frame_container, text="Descripcion de referencia", font=ctk.CTkFont(size=12, weight='bold'))
        self.textbox_description=ctk.CTkTextbox(master=self.frame_container, width=350, wrap='word')
        button_save_metadata=ctk.CTkButton(master=self.frame_container, text="Guardar", fg_color="green2", hover_color="green3", command=self.save_metadata)
        self.var_count=ctk.StringVar(value="0")
        label_count=ctk.CTkLabel(master=self.frame_container, textvariable=self.var_count, font=ctk.CTkFont(size=25, weight='bold'))
        self.frame_container.insert_element(cad_pos="0,0", element=button_reset, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="1,0", element=label_folder_name, padx=5, pady=1, sticky="")
        self.frame_container.insert_element(cad_pos="2,0", element=entry_folder_name, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="3,0", element=label_file_name, padx=5, pady=1, sticky="")
        self.frame_container.insert_element(cad_pos="4,0", element=entry_file_name, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="5,0", element=check_box_trajectory, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="6,0", element=label_name, padx=5, pady=1, sticky="")
        self.frame_container.insert_element(cad_pos="7,0", element=entry_name, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="8,0", element=label_description, padx=5, pady=1, sticky="")
        self.frame_container.insert_element(cad_pos="9,0", element=self.textbox_description, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="10,0", element=button_save_metadata, padx=5, pady=5, sticky="")
        self.frame_container.insert_element(cad_pos="11,0", element=label_count, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=check_box_save_estimations, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=self.frame_container, padx=5, pady=5, sticky="").hide_frame()

    # Para detectar cuando se activa el 'checkbox'
    def trace_var(self, var, index, mode):
        if self.var_save_estimations.get():
            self.frame_container.show_frame()
            self.thread_keyboard=ThreadKeyboard(callback=self.key_pressed)
            self.thread_keyboard.start()
        else:
            self.frame_container.hide_frame()
            if self.thread_keyboard is not None:
                self.keyboard_controller.release(keyboard.Key.esc)
                self.thread_keyboard=None

    # Para reiniciar todas las variables
    def reset_all(self):
        self.var_folder_name.set(value='')
        self.var_file_name.set(value='')
        self.var_trajectory.set(value=0)
        self.var_name.set(value="")
        self.textbox_description.delete("0.0", "end")
        self.var_count.set(value='0')

    # Para guardar los algoritmos
    def set_algorithm(self, algorithm):
        self.algorithm=algorithm

    # Para guardar los metadatos de lo que se capturo
    def save_metadata(self):
        name=self.var_name.get()
        description=self.textbox_description.get("0.0", "end")
        d={
            "name": name,
            "description": description
        }
        corresponding_folder_name="trajectory_datasets" if self.var_trajectory.get() else "datasets"
        path=os.path.join(self.app.folder_name_data, *[corresponding_folder_name, self.var_folder_name.get()])
        # Si el directorio no existe entonces se crea
        if not os.path.exists(path):
            os.makedirs(path)
        path=os.path.join(path, *[f'{self.var_folder_name.get()}.pkl'])
        with open(path, "wb") as f:
            pickle.dump(d, f)

    # Para guardar las estimaciones 3D cuando se haya presionado un boton
    def key_pressed(self, key):
        try:
            if (self.var_folder_name.get() != '' and self.var_file_name.get() != '') and (str(key.name) == 'media_volume_up' or str(key.name) == 'media_volume_down'):
                corresponding_folder_name="trajectory_datasets" if self.var_trajectory.get() else "datasets"
                path=os.path.join(self.app.folder_name_data, *[corresponding_folder_name, self.var_folder_name.get()])
                # Si el directorio no existe entonces se crea
                if not os.path.exists(path):
                    os.makedirs(path)
                # Se continua en la numeracion del ultimo dato guardado
                files=None
                if corresponding_folder_name == "trajectory_datasets":
                    files=list(filter(lambda elem: elem.split("_")[0] == "data" and elem.split(".")[-1] == "npy", os.listdir(path)))
                else:
                    files=list(filter(lambda elem: "_".join(elem.split("_")[0:2 if self.var_folder_name.get() == "all_postures" else 3]) == self.var_folder_name.get() and elem.split(".")[-1] == "npy", os.listdir(path)))
                self.var_count.set(value=len(files))
                
                if str(key.name) == 'media_volume_up':
                    # print("Volume Up")
                    if self.algorithm is not None:
                        if self.algorithm.points_3D is not None and self.algorithm.classification_euler_angles is not None:
                            self.var_count.set(value=str(int(self.var_count.get()) + 1))
                            # Se guardan los puntos 3D de la postura actual
                            np.save(os.path.join(path, *[f"{self.var_file_name.get()}_{self.var_count.get().zfill(4)}.npy"]), self.algorithm.points_3D)   
                            if corresponding_folder_name == "datasets":
                                # Se guardan los angulos de Euler para la clasificacion de la postura actual
                                np.save(os.path.join(path, *[f"euler_angles_for_classification_{self.var_count.get().zfill(4)}.npy"]), self.algorithm.classification_euler_angles)
                if str(key.name) == 'media_volume_down':
                    # print("Volume Down")
                    pass
        except AttributeError:
            pass