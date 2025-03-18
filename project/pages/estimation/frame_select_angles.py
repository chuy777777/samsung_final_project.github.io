import customtkinter  as ctk
import tkinter as tk
import numpy as np
import os

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.text_validator import TextValidator
import components.utils as utils

class FrameSelectAngles(CreateFrame):
    def __init__(self, master, name, algorithm_name, number_points, callback=None, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(5,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.folder_name_algorithm_images=self.app.folder_name_algorithm_images
        self.algorithm_name=algorithm_name
        self.number_points=number_points
        self.callback=callback
        self.triplet_list=[]

        label_algorithm_name=ctk.CTkLabel(master=self, text=self.algorithm_name)
        label_algorithm_image=ctk.CTkLabel(master=self, text="", image=utils.image_to_ctk_img(image_full_path=os.path.join(self.folder_name_algorithm_images, *["{}.png".format(self.algorithm_name)]), width=300, height=300))
        
        frame_inputs=CreateFrame(master=self, grid_frame=GridFrame(dim=(4,3), arr=np.array([["0,0","0,0","0,0"],["1,0","1,1","1,2"],["2,0","2,1","2,2"],["3,0","3,0","3,0"]])))
        label_angles=ctk.CTkLabel(master=frame_inputs, text="Angulos")
        label_p1=ctk.CTkLabel(master=frame_inputs, text="Punto 1")
        label_pm=ctk.CTkLabel(master=frame_inputs, text="Punto medio")
        label_p2=ctk.CTkLabel(master=frame_inputs, text="Punto 2")
        self.var_p1=ctk.StringVar(value="")
        entry_p1=ctk.CTkEntry(master=frame_inputs, width=70, textvariable=self.var_p1)
        self.var_pm=ctk.StringVar(value="")
        entry_pm=ctk.CTkEntry(master=frame_inputs, width=70, textvariable=self.var_pm)
        self.var_p2=ctk.StringVar(value="")
        entry_p2=ctk.CTkEntry(master=frame_inputs, width=70, textvariable=self.var_p2)
        button_add=ctk.CTkButton(master=frame_inputs, text="Agregar", fg_color="green2", hover_color="green3", command=self.add_triplet)
        frame_inputs.insert_element(cad_pos="0,0", element=label_angles, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="1,0", element=label_p1, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="1,1", element=label_pm, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="1,2", element=label_p2, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="2,0", element=entry_p1, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="2,1", element=entry_pm, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="2,2", element=entry_p2, padx=5, pady=5, sticky="")
        frame_inputs.insert_element(cad_pos="3,0", element=button_add, padx=5, pady=5, sticky="")

        frame_delete=CreateFrame(master=self, grid_frame=GridFrame(dim=(2,2), arr=np.array([["0,0","0,0"],["1,0","1,1"]])))
        self.var_triplet=ctk.StringVar(value="")
        button_delete_all=ctk.CTkButton(master=frame_delete, text="Eliminar todo", fg_color="red2", hover_color="red3", command=self.delete_all_triplet)
        entry_triplet=ctk.CTkEntry(master=frame_delete, width=70, textvariable=self.var_triplet, placeholder_text="p1,pm,p2")
        button_delete=ctk.CTkButton(master=frame_delete, text="Eliminar", fg_color="red2", hover_color="red3", command=self.delete_triplet)
        frame_delete.insert_element(cad_pos="0,0", element=button_delete_all, padx=5, pady=5, sticky="")
        frame_delete.insert_element(cad_pos="1,0", element=entry_triplet, padx=5, pady=5, sticky="")
        frame_delete.insert_element(cad_pos="1,1", element=button_delete, padx=5, pady=5, sticky="")

        self.var_label_information=ctk.StringVar(value="")
        label_information=ctk.CTkLabel(master=self, textvariable=self.var_label_information, wraplength=300)

        self.insert_element(cad_pos="0,0", element=label_algorithm_name, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=label_algorithm_image, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=frame_inputs, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=frame_delete, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="4,0", element=label_information, padx=5, pady=5, sticky="ew")

    def is_valid_point(self, p):
        return p >= 0 and p < self.number_points
    
    def exists_in_current_triplet_list(self, p1, pm, p2):
        for i in range(len(self.triplet_list)):
            triplet=self.triplet_list[i]
            if triplet[0] == p1 and triplet[1] == pm and triplet[2] == p2:
                return True,i 
        return False,-1

    def validate_triplet(self, text_p1, text_pm, text_p2):
        p1=TextValidator.validate_number(text=text_p1)
        pm=TextValidator.validate_number(text=text_pm)
        p2=TextValidator.validate_number(text=text_p2)
        if p1 is not None and pm is not None and p2 is not None:
            p1,pm,p2=int(p1),int(pm),int(p2)
            if self.is_valid_point(p=p1) and self.is_valid_point(p=pm) and self.is_valid_point(p=p2):
                if p1 == pm or p1 == p2 or pm == p2:
                    return None,None,None,False
                return p1,pm,p2,True
        return None,None,None,False

    def update_information(self):
        text=' '.join(list(map(lambda elem: "({},{},{})".format(elem[0],elem[1],elem[2]),self.triplet_list)))
        self.var_label_information.set(value=text)

    def add_triplet(self):
        text_p1,text_pm,text_p2=self.var_p1.get().strip(),self.var_pm.get().strip(),self.var_p2.get().strip()
        p1,pm,p2,is_valid=self.validate_triplet(text_p1=text_p1, text_pm=text_pm, text_p2=text_p2)
        if is_valid:
            exists,index=self.exists_in_current_triplet_list(p1=p1, pm=pm, p2=p2) 
            if not exists:
                self.triplet_list.append((p1,pm,p2))
                self.update_information()
                if self.callback is not None:
                    self.callback()
            else:
                tk.messagebox.showinfo(title="Agregar angulo", message="La tripleta de numeros no es valida\n\nPosibles problemas:\n\n - La tripleta ya existe\n - Los valores no pueden ser iguales\n - Los valores no estan en el rango [0,{}]".format(self.number_points - 1))
        else:
            tk.messagebox.showinfo(title="Agregar angulo", message="La tripleta de numeros no es valida\n\nPosibles problemas:\n\n - La tripleta ya existe\n - Los valores no pueden ser iguales\n - Los valores no estan en el rango [0,{}]".format(self.number_points - 1))
        
    def delete_all_triplet(self):
        self.triplet_list=[]
        self.update_information()
        if self.callback is not None:
            self.callback()

    def delete_triplet(self):
        split=self.var_triplet.get().split(",") 
        if len(split) == 3:
            text_p1,text_pm,text_p2=split[0].strip(),split[1].strip(),split[2].strip()
            p1,pm,p2,is_valid=self.validate_triplet(text_p1=text_p1, text_pm=text_pm, text_p2=text_p2)
            if is_valid:
                exists,index=self.exists_in_current_triplet_list(p1=p1, pm=pm, p2=p2) 
                if exists:
                    self.triplet_list.pop(index)
                    self.update_information()
                    if self.callback is not None:
                        self.callback()
                else:
                    tk.messagebox.showinfo(title="Eliminar angulo", message="La tripleta de numeros no es valida")
            else:
                tk.messagebox.showinfo(title="Eliminar angulo", message="La tripleta de numeros no es valida")
        else:
            tk.messagebox.showinfo(title="Eliminar angulo", message="Debe ingresar una tripleta de numeros separados por coma")
