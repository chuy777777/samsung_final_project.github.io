import numpy as np
import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.thread_camera import ThreadCameraFrameListener
from components.frame_graphic_2D import FrameGraphic2D
from components.frame_graphic_3D import FrameGraphic3D

"""
Frame para mostrar toda la informacion del proceso de calibracion de la camara.
"""
class FrameCalibrationInformation(CreateFrame, ThreadCameraFrameListener):
    def __init__(self, master, name, thread_camera, only_camera_parameters=False, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(3,1), arr=None), **kwargs)
        ThreadCameraFrameListener.__init__(self)
        self.thread_camera=thread_camera
        # Para escuchar notificaciones cuando la camara cambie
        self.thread_camera.add_frame_listener(frame_listener=self)
        self.only_camera_parameters=only_camera_parameters

    def destroy(self):
        self.thread_camera.delete_frame_listener(frame_listener_name=self.name)
        CreateFrame.destroy(self)

    # Metodo para la escucha de notificaciones cuando la camara cambie
    def thread_camera_frame_listener_notification(self):
        self.destroy_all()
        if self.thread_camera.camera_device.calibration_information_is_loaded:
            self.show_calibration_information()
        else:
            self.show_not_calibration_information()

    # Para mostrar la informacion del proceso de calibracion
    def show_calibration_information(self):
        camera_device=self.thread_camera.camera_device

        # Solo mostrar los parametros de la camara
        if self.only_camera_parameters:
            label_camera_parameters=ctk.CTkLabel(master=self, text=camera_device.__str__())
            self.insert_element(cad_pos="0,0", element=label_camera_parameters, padx=5, pady=5, sticky="ew")
        # Mostrar los parametros de la camara y los resultados del proceso de calibracion
        else:
            label_camera_parameters=ctk.CTkLabel(master=self, text=camera_device.__str__())
            # Grafico 2D
            frame_graphic_2D=FrameGraphic2D(master=self, name="FrameGraphic2D_{}".format(camera_device.camera_name), width=400, height=400)
            ps=np.concatenate([np.arange(1,camera_device.history_L.size + 1)[:,None], camera_device.history_L[:,None]], axis=1)
            # Graficar el historial de perdida de la funcion objetivo
            frame_graphic_2D.plot_lines(ps=ps, color_rgb=(0,0,255))
            # Nombre de las etiquetas del grafico
            frame_graphic_2D.graphic_configuration(xlabel="Iteraciones", ylabel="L", title="Historial de perdida")
            # Actualizar grafico
            frame_graphic_2D.draw()
            # Grafico 3D (las unidades se manejan en metros)
            frame_graphic_3D=FrameGraphic3D(master=self, name="FrameGraphic3D_{}".format(camera_device.camera_name), square_size=0.7, width=400, height=400)
            # Vector de traslacion y matriz de rotacion del sistema de coordenadas de la camara con respecto al sistema de coordenadas tradicional (matplotlib)
            tcm,Tcm=np.array([[0],[-0.5],[0]]),np.array([[1,0,0],[0,0,1],[0,-1,0]])
            # Mostrar sistema de coordenadas de la camara 
            frame_graphic_3D.plot_coordinate_system(t_list=[tcm], T_list=[Tcm], length=0.5)
            # Factor de conversion de milimetros a metros
            factor_mm_to_mt=1 / 1000
            # Se mostraran todos los sistemas de coordenadas del mundo de cada matriz extrinseca
            for i in range(camera_device.Qs.shape[0]):
                # Matriz extrinseca
                Q=camera_device.Qs[i]
                # Vector de traslacion y matriz de rotacion
                twc,Twc=Q[:,[3]],Q[:,[0,1,2]]
                # Mostrar sistema de coordenadas del mundo
                H=frame_graphic_3D.plot_coordinate_system(t_list=[tcm,twc * factor_mm_to_mt], T_list=[Tcm,Twc], length=0.5)
                # Longitud del plano a dibujar
                length=0.5
                # Vertices del plano a dibujar
                vm_=H @ np.array([[0,0,0,1],[length,0,0,1],[length,length,0,1],[0,length,0,1]]).T
                # Puntos en forma de vector fila
                vm=vm_[0:3,:].T
                # Mostrar plano XY en el sistema de coordenadas del mundo
                frame_graphic_3D.plot_polygon(verts=vm, alpha=0.8, facecolors="orange", edgecolors="black")
            # Titulo del grafico
            frame_graphic_3D.set_title(title="Matrices extrinsecas Q's ({})\nEjes: X (rojo) Y (verde) Z (azul)".format(camera_device.Qs.shape[0]))
            # Actualizar grafico
            frame_graphic_3D.draw()

            self.insert_element(cad_pos="0,0", element=label_camera_parameters, padx=5, pady=5, sticky="ew")
            self.insert_element(cad_pos="1,0", element=frame_graphic_2D, padx=5, pady=5, sticky="")
            self.insert_element(cad_pos="2,0", element=frame_graphic_3D, padx=5, pady=5, sticky="")

    # Para mostrar cuando no exista informacion de la calibracion de la camara (la camara aun no esta calibrada)
    def show_not_calibration_information(self):
        label_not_camera_parameters=ctk.CTkLabel(master=self, text=" --- CAMARA NO CALIBRADA ---")

        self.insert_element(cad_pos="0,0", element=label_not_camera_parameters, padx=5, pady=5, sticky="ew")
