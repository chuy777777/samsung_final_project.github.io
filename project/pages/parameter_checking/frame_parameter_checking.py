from components.create_frame import CreateFrame
from components.create_scrollable_frame import CreateScrollableFrame
from components.grid_frame import GridFrame
from components.frame_camera_display import FrameCameraDisplay
from pinhole_camera_model.augmented_reality import AugmentedReality
from pages.parameter_checking.frame_calibration_information import FrameCalibrationInformation

"""
Frame para mostrar el correcto funcionamiento de los parametros de la camara obtenidos en el proceso de calibracion.
"""
class FrameParameterChecking(CreateScrollableFrame):
    def __init__(self, master, name, **kwargs):
        CreateScrollableFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)
        self.app=self.get_frame(frame_name="FrameApplication") 
        self.thread_camera_1=self.app.thread_camera_1
        self.thread_camera_2=self.app.thread_camera_2
        self.rate_ms=100

        # Mostrar informacion de la calibracion de las camaras
        frame_calibration_informations=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        frame_calibration_information_camera_1=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation1", thread_camera=self.thread_camera_1, only_camera_parameters=False)
        frame_calibration_information_camera_2=FrameCalibrationInformation(master=frame_calibration_informations, name="FrameCalibrationInformation2", thread_camera=self.thread_camera_2, only_camera_parameters=False)
        frame_calibration_informations.insert_element(cad_pos="0,0", element=frame_calibration_information_camera_1, padx=5, pady=5, sticky="")
        frame_calibration_informations.insert_element(cad_pos="0,1", element=frame_calibration_information_camera_2, padx=5, pady=5, sticky="")

        # Visualizacion de las camaras
        frame_camera_displays=CreateFrame(master=self, grid_frame=GridFrame(dim=(1,2), arr=None))
        self.frame_camera_display_camera_1=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay1", thread_camera=self.thread_camera_1, rate_ms=self.rate_ms, scale_percent=70, editable=True)
        self.frame_camera_display_camera_2=FrameCameraDisplay(master=frame_camera_displays, name="FrameCameraDisplay2", thread_camera=self.thread_camera_2, rate_ms=self.rate_ms, scale_percent=70, editable=True)
        frame_camera_displays.insert_element(cad_pos="0,0", element=self.frame_camera_display_camera_1, padx=5, pady=5, sticky="")
        frame_camera_displays.insert_element(cad_pos="0,1", element=self.frame_camera_display_camera_2, padx=5, pady=5, sticky="")

        self.insert_element(cad_pos="0,0", element=frame_calibration_informations, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=frame_camera_displays, padx=5, pady=5, sticky="ew")

        # Realidad aumentada utilizando los parametros de las camaras para verificar su correcto funcionamiento
        self.augmented_reality()

    # Para mostrar un cubo en realidad aumentada utilizando los parametros de la camara
    def augmented_reality(self):
        camera_device_1=self.thread_camera_1.camera_device
        camera_device_2=self.thread_camera_2.camera_device
        if camera_device_1.calibration_information_is_loaded:
            self.frame_camera_display_camera_1.update_label_camera(frame_bgr=AugmentedReality.draw_cube_on_chessboard(frame_bgr=self.thread_camera_1.frame_bgr, chessboard_dimensions=camera_device_1.calibration_parameters.chessboard_dimensions, square_size=camera_device_1.calibration_parameters.square_size, K=camera_device_1.K, K_inv=camera_device_1.K_inv, q=camera_device_1.q, fast=True))
        else:
            self.frame_camera_display_camera_1.update_label_camera(frame_bgr=self.thread_camera_1.frame_bgr)
        if camera_device_2.calibration_information_is_loaded:
            self.frame_camera_display_camera_2.update_label_camera(frame_bgr=AugmentedReality.draw_cube_on_chessboard(frame_bgr=self.thread_camera_2.frame_bgr, chessboard_dimensions=camera_device_2.calibration_parameters.chessboard_dimensions, square_size=camera_device_2.calibration_parameters.square_size, K=camera_device_2.K, K_inv=camera_device_2.K_inv, q=camera_device_2.q, fast=True))
        else:
            self.frame_camera_display_camera_2.update_label_camera(frame_bgr=self.thread_camera_2.frame_bgr)

        self.after(self.rate_ms, self.augmented_reality)  