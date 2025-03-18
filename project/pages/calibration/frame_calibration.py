import numpy as np
import customtkinter  as ctk
import tkinter as tk

from components.create_frame import CreateFrame
from components.create_scrollable_frame import CreateScrollableFrame
from components.grid_frame import GridFrame
from components.frame_camera_display import FrameCameraDisplay
import components.utils as utils
from pages.parameter_checking.frame_calibration_information import FrameCalibrationInformation
from pages.calibration.frame_calibration_parameters import FrameCalibrationParameters
from pinhole_camera_model.image_points_2D_3D import ImagePoints2D3D
from pinhole_camera_model.draw_image import DrawImage
from pinhole_camera_model.optimization_parameters import OptimizationParameters
from pinhole_camera_model.thread_calibration_process import ThreadCalibrationProcess
        
"""
Frame para el proceso de calibracion de la camara.
"""
class FrameCalibration(CreateScrollableFrame):
    def __init__(self, master, name, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(4,2), arr=np.array([["0,0","0,0"],["1,0","1,0"],["2,0","2,1"],["3,0","3,0"]])), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        self.rate_ms=100
        self.optimization_parameters=OptimizationParameters(num_it=100, lr=0.001, beta_1=0.9, beta_2=0.999)

        # Mostrar informacion de la calibracion de las camaras
        frame_calibration_informations=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        frame_calibration_information_camera_1=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation1", thread_camera=self.thread_camera_1, only_camera_parameters=True)
        frame_calibration_information_camera_2=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation2", thread_camera=self.thread_camera_2, only_camera_parameters=True)
        frame_calibration_informations.insert_element(cad_pos="0,0", element=frame_calibration_information_camera_1, padx=5, pady=5, sticky="")
        frame_calibration_informations.insert_element(cad_pos="0,1", element=frame_calibration_information_camera_2, padx=5, pady=5, sticky="")

        # Visualizacion de las camaras
        frame_camera_displays=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        self.frame_camera_display_camera_1=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay1", thread_camera=self.thread_camera_1, rate_ms=self.rate_ms, scale_percent=50, editable=False)
        self.frame_camera_display_camera_2=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay2", thread_camera=self.thread_camera_2, rate_ms=self.rate_ms, scale_percent=50, editable=False)
        frame_camera_displays.insert_element(cad_pos="0,0", element=self.frame_camera_display_camera_1, padx=5, pady=5, sticky="")
        frame_camera_displays.insert_element(cad_pos="0,1", element=self.frame_camera_display_camera_2, padx=5, pady=5, sticky="")

        # Componentes de configuracion para el proceso de calibracion
        frame_container=CreateFrame(master=self, grid_frame=GridFrame(dim=(5,2), arr=np.array([["0,0","0,0"],["1,0","1,0"],["2,0","2,1"],["3,0","3,0"],["4,0","4,0"]])))
        self.var_camera_selected=ctk.StringVar(value="Camara 1")
        label_select_camera=ctk.CTkLabel(master=frame_container, text="Selecciona la camara a calibrar")
        option_menu_select_camera=ctk.CTkOptionMenu(master=frame_container, values=["Camara 1","Camara 2"], variable=self.var_camera_selected)
        button_start=ctk.CTkButton(master=frame_container, text="Iniciar", fg_color="green2", hover_color="green3", command=self.start_take_photos)
        button_stop=ctk.CTkButton(master=frame_container, text="Detener", fg_color="red2", hover_color="red3", command=self.stop_take_photos)
        self.var_timer=ctk.StringVar(value="")
        label_timer=ctk.CTkLabel(master=frame_container, textvariable=self.var_timer)
        self.var_message=ctk.StringVar(value="")
        label_message=ctk.CTkLabel(master=frame_container, textvariable=self.var_message)
        frame_container.insert_element(cad_pos="0,0", element=label_select_camera, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="1,0", element=option_menu_select_camera, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="2,0", element=button_start, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="2,1", element=button_stop, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="3,0", element=label_timer, padx=5, pady=5, sticky="")
        frame_container.insert_element(cad_pos="4,0", element=label_message, padx=5, pady=5, sticky="")
        
        self.frame_calibration_parameters=FrameCalibrationParameters(master=self, name="FrameCalibrationParameters")

        self.insert_element(cad_pos="0,0", element=frame_calibration_informations, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=frame_camera_displays, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=frame_container, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="3,0", element=self.frame_calibration_parameters, padx=5, pady=5, sticky="")

        self.init_calibration_variables()

    # Para inicializar las variables del proceso de calibracion
    def init_calibration_variables(self, thread_camera_selected=None, calibration_parameters=None):
        self.thread_camera_selected=thread_camera_selected
        self.calibration_parameters=calibration_parameters
        self.band_stop_take_photos=False
        self.good_images_list=[]
        self.thread_calibration_process=None
        self.var_timer.set(value="")
        self.var_message.set(value="")

    # Para iniciar la captura de imagenes del tablero de ajedrez en diferentes posiciones y orientaciones
    def start_take_photos(self):
        camera_selected=self.var_camera_selected.get()
        thread_camera_selected=self.thread_camera_1 if camera_selected == "Camara 1" else self.thread_camera_2
        if thread_camera_selected.camera_device.camera_name != "":
            calibration_parameters=self.frame_calibration_parameters.get_calibration_parameters()
            if calibration_parameters is not None:
                if tk.messagebox.askyesnocancel(title="Calibracion de camara", message="¿Esta seguro de calibrar '{}' ({})?\n\nA continuacion se tomaran fotos del tablero de ajedrez para la calibracion.".format(camera_selected, thread_camera_selected.camera_device.camera_name)):
                    self.init_calibration_variables(thread_camera_selected=thread_camera_selected, calibration_parameters=calibration_parameters)
                    self.destroy_element(cad_pos="2,1")
                    self.insert_element(cad_pos="2,1", element=CreateScrollableFrame(master=self, name="FramePhotos", grid_frame=GridFrame(dim=(1,int(calibration_parameters.S)), arr=None), orientation="horizontal", height=250), padx=5, pady=5, sticky="ew")
                    self.photo_counter(index=0, t=calibration_parameters.timer_time)
                else:
                    self.init_calibration_variables()
            else:
                self.init_calibration_variables()
                tk.messagebox.showinfo(title="Calibracion de camara", message="Los parametros de calibracion no estan bien escritos.")
        else:
            self.init_calibration_variables()
            tk.messagebox.showinfo(title="Calibracion de camara", message="Debe seleccionar primero una camara.")

    # Para detener la captura de imagenes 
    def stop_take_photos(self):
        self.band_stop_take_photos=True

    # Para tomar una imagen cada cierto tiempo y un cierto numero de veces
    def photo_counter(self, index, t):
        if self.band_stop_take_photos:
            self.init_calibration_variables()
            self.destroy_element(cad_pos="2,1")
            return
        if index < self.calibration_parameters.S:
            if t > 0:
                conv_factor=1 / 1000   # ms a s 
                t-=self.rate_ms * conv_factor
                if t < 0:
                    t=0
                self.var_timer.set(value="Foto '{}' de '{}': '{}'".format(index + 1, int(self.calibration_parameters.S), round(t, 2)))
            else:
                index+=1
                t=self.calibration_parameters.timer_time
                self.capture_photo(index=index)
            self.after(self.rate_ms, lambda: self.photo_counter(index=index, t=t))
        else:
            self.var_timer.set(value="")
            self.start_calibration_process()

    # Para capturar la imagen,  guardar la imagen, y obtener los puntos 2D de las esquinas del tablero de ajedrez
    def capture_photo(self, index):
        label_photo=ctk.CTkLabel(master=self.get_element(cad_pos="2,1"), text="")
        frame_bgr=self.thread_camera_selected.frame_bgr
        if frame_bgr is not None:
            v3pds,vws,is_ok=ImagePoints2D3D.point_pairs_2D_and_3D_chessboard(frame_bgr=frame_bgr, chessboard_dimensions=self.calibration_parameters.chessboard_dimensions, square_size=self.calibration_parameters.square_size, fast=False)
            if is_ok:
                self.good_images_list.append((frame_bgr,v3pds,vws))
                frame_bgr=DrawImage.draw_points(frame_bgr=frame_bgr, points=v3pds, size=5, color_bgr=(0,0,255))
            else:
                frame_bgr[:,:,0]=255
            frame_bgr=utils.resize_frame_bgr(scale_percent=50, frame_bgr=frame_bgr)
        img=utils.frame_bgr_to_ctk_img(frame_bgr=frame_bgr)
        label_photo.configure(image=img)
        label_photo.image=img
        self.get_element(cad_pos="2,1").insert_element(cad_pos="0,{}".format(int(self.calibration_parameters.S - index)), element=label_photo, padx=5, pady=5, sticky="")

    # Para comenzar el proceso de obtencion de los parametros de la camara 
    def start_calibration_process(self):
        if len(self.good_images_list) >= 3:
            self.var_message.set(value="Espere mientras se lleva a cabo la calibracion...")
            self.thread_calibration_process=ThreadCalibrationProcess(thread_camera=self.thread_camera_selected, v3pds_list=list(map(lambda elem: elem[1], self.good_images_list)), vws=self.good_images_list[0][2], calibration_parameters=self.calibration_parameters, optimization_parameters=self.optimization_parameters)
            self.thread_calibration_process.start()
            self.wait_thread_calibration_process()
        else:
            tk.messagebox.showinfo(title="Proceso de calibracion", message="No se capturaron las suficientes imagenes buenas.")
    
    # Para esperar a que termine el proceso de obtencion de los parametros de la camara 
    def wait_thread_calibration_process(self):
        if self.thread_calibration_process.is_alive():
            self.after(self.rate_ms, self.wait_thread_calibration_process)
        else:
            self.var_message.set(value="")
            if self.thread_calibration_process.is_ok:
                tk.messagebox.showinfo(title="Proceso de calibracion", message="La camara '{}' se ha calibrado correctamente.".format(self.thread_camera_selected.camera_device.camera_name))
            else:
                tk.messagebox.showinfo(title="Proceso de calibracion", message="No fue exitosa la calibracion para la camara '{}'.\nTomar mas imagenes en posiciones y orientaciones diferentes puede ayudar.".format(self.thread_camera_selected.camera_device.camera_name))

        
