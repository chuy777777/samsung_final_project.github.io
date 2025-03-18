import numpy as np
import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame

"""
Frame para mostrar la clasificacion de la postura (las predicciones para cada clase)
"""
class FrameShowPostureClassification(CreateFrame):
    def __init__(self, master, name, title, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(3,2), arr=np.array([["0,0","0,0"],["1,0","1,1"],["2,0","2,1"]])), **kwargs)
         
        label_title=ctk.CTkLabel(master=self, text=title, font=ctk.CTkFont(size=12, weight='bold'))
        label_class_0=ctk.CTkLabel(master=self, text="Clase 0", font=ctk.CTkFont(size=12, weight='bold'))
        label_class_1=ctk.CTkLabel(master=self, text="Clase 1", font=ctk.CTkFont(size=12, weight='bold'))
        self.var_class_0=ctk.StringVar(value="")
        entry_class_0=ctk.CTkEntry(master=self, state="disabled", width=50, textvariable=self.var_class_0)
        self.var_class_1=ctk.StringVar(value="")
        entry_class_1=ctk.CTkEntry(master=self, state="disabled", width=50, textvariable=self.var_class_1)
        
        self.insert_element(cad_pos="0,0", element=label_title, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=label_class_0, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,1", element=label_class_1, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,0", element=entry_class_0, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="2,1", element=entry_class_1, padx=5, pady=5, sticky="")

    # Para actualizar el texto de las predicciones para cada clase
    def update_prediction(self, pred):
        # NOTA: La suma de ambas probabilidades es 1
        # Prediccion para la clase 0 (clase negativa)
        pred_class_0=pred[0]
        # Prediccion para la clase 1 (clase positiva)
        pred_class_1=pred[1]
        self.var_class_0.set(value=f"{pred_class_0:.2f}")
        self.var_class_1.set(value=f"{pred_class_1:.2f}")