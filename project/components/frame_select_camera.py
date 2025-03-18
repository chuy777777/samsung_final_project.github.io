import numpy as np
import customtkinter  as ctk
import subprocess
import re

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame

"""
Frame para elegir las camaras a utilizar.
"""
class FrameSelectCamera(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(5,1), arr=np.array([["0,0"],["1,0"],["2,0"],["3,0"],["4,0"]])), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        
        self.camera_name_none=""
        self.var_camera_name_1=ctk.StringVar(value=self.camera_name_none)
        self.var_camera_name_2=ctk.StringVar(value=self.camera_name_none)
        camera_information_list=self.get_camera_information_list()
        camera_names=[self.camera_name_none] + list(map(lambda elem: elem[0], camera_information_list))

        button_reload=ctk.CTkButton(master=self, text="Recargar", command=self.reload)
        label_camera_1=ctk.CTkLabel(master=self, text="Camara 1")
        option_menu_camera_1=ctk.CTkOptionMenu(master=self, values=camera_names, variable=self.var_camera_name_1, command=lambda camera_name: self.set_camera(root_name="camera_1", camera_name=camera_name))
        label_camera_2=ctk.CTkLabel(master=self, text="Camara 2")
        option_menu_camera_2=ctk.CTkOptionMenu(master=self, values=camera_names, variable=self.var_camera_name_2, command=lambda camera_name: self.set_camera(root_name="camera_2", camera_name=camera_name))

        self.insert_element(cad_pos="0,0", element=button_reload, padx=5, pady=1)
        self.insert_element(cad_pos="1,0", element=label_camera_1, padx=5, pady=1)
        self.insert_element(cad_pos="2,0", element=option_menu_camera_1, padx=5, pady=1)
        self.insert_element(cad_pos="3,0", element=label_camera_2, padx=5, pady=1)
        self.insert_element(cad_pos="4,0", element=option_menu_camera_2, padx=5, pady=1)

    # Para buscar y mostrar las camaras conectadas via USB
    def reload(self):
        camera_information_list=self.get_camera_information_list()
        camera_names=[self.camera_name_none] + list(map(lambda elem: elem[0], camera_information_list))
        self.get_element(cad_pos="2,0").configure(values=camera_names)
        self.get_element(cad_pos="4,0").configure(values=camera_names)
        if self.thread_camera_1.camera_device.camera_name not in camera_names:
            self.thread_camera_1.camera_device.init_device()
            self.var_camera_name_1.set(value=self.camera_name_none)
        if self.thread_camera_2.camera_device.camera_name not in camera_names:
            self.thread_camera_2.camera_device.init_device()
            self.var_camera_name_2.set(value=self.camera_name_none)

    # Para obtener informacion de las camaras conectadas via USB
    def get_camera_information_list(self):
        camera_information_list=[]
        res=subprocess.run(["v4l2-ctl", "--list-devices"], capture_output=True, text=True)
        res_split=res.stdout.split("\n\n")[0:-1]
        pattern = "video[0-9]{1}"
        for r in res_split:
            r_split=r.split("\n\t")
            if len(re.findall(pattern, r_split[1]))>0:
                camera_name=r_split[0].split("(")[0].strip() 
                camera_path=r_split[1]
                camera_information_list.append((camera_name, camera_path))
        return camera_information_list
    
    # Para seleccionar una camara 
    def set_camera(self, root_name, camera_name):
        camera_information_list=self.get_camera_information_list()
        if root_name == "camera_1":
            if camera_name == self.camera_name_none:
                self.thread_camera_1.init_cap()
            else:
                camera_name,camera_path=list(filter(lambda elem: elem[0] == camera_name,camera_information_list))[0]
                if camera_name != self.thread_camera_1.camera_device.camera_name:
                    if camera_name != self.thread_camera_2.camera_device.camera_name:
                        self.thread_camera_1.init_cap(camera_name=camera_name, camera_path=camera_path)
                    else:
                        self.var_camera_name_1.set(value=self.thread_camera_1.camera_device.camera_name)
        elif root_name == "camera_2":
            if camera_name == self.camera_name_none:
                self.thread_camera_2.init_cap()
            else:
                camera_name,camera_path=list(filter(lambda elem: elem[0] == camera_name,camera_information_list))[0]
                if camera_name != self.thread_camera_2.camera_device.camera_name:
                    if camera_name != self.thread_camera_1.camera_device.camera_name:
                        self.thread_camera_2.init_cap(camera_name=camera_name, camera_path=camera_path)
                    else:
                        self.var_camera_name_2.set(value=self.thread_camera_2.camera_device.camera_name)
