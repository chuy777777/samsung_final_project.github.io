import customtkinter  as ctk

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.text_validator import TextValidator
from pinhole_camera_model.calibration_parameters import CalibrationParameters

"""
Frame para mostrar un formulario de los parametros de calibracion.
"""
class FrameCalibrationParameters(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(4,2), arr=None), **kwargs)
        
        self.var_chessboard_dimensions=ctk.StringVar(value="9,6")
        self.var_square_size=ctk.StringVar(value="25.0")
        self.var_S=ctk.StringVar(value="5")
        self.var_timer_time=ctk.StringVar(value="3")
        label_chessboard_dimensions=ctk.CTkLabel(master=self, text="Dimensiones tablero (w,h)\nEjemplo: 9,6")
        label_square_size=ctk.CTkLabel(master=self, text="Dimensiones cuadrado (mm)")
        label_S=ctk.CTkLabel(master=self, text="Numero de imagenes\n(minimo 3)")
        label_timer_time=ctk.CTkLabel(master=self, text="Tiempo entre captura (s)")
        entry_chessboard_dimensions=ctk.CTkEntry(master=self, textvariable=self.var_chessboard_dimensions)
        entry_square_size=ctk.CTkEntry(master=self, textvariable=self.var_square_size)
        entry_S=ctk.CTkEntry(master=self, textvariable=self.var_S)
        entry_timer_time=ctk.CTkEntry(master=self, textvariable=self.var_timer_time)

        self.insert_element(cad_pos="0,0", element=label_chessboard_dimensions, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="0,1", element=label_square_size, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,0", element=entry_chessboard_dimensions, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="1,1", element=entry_square_size, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,0", element=label_S, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="2,1", element=label_timer_time, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,0", element=entry_S, padx=5, pady=5, sticky="ew")
        self.insert_element(cad_pos="3,1", element=entry_timer_time, padx=5, pady=5, sticky="ew")

    # Para obtener los parametros de calibracion del formulario
    def get_calibration_parameters(self):
        chessboard_dimensions=TextValidator.validate_tuple2(text=self.var_chessboard_dimensions.get())
        square_size=TextValidator.validate_number(text=self.var_square_size.get())
        S=TextValidator.validate_number(text=self.var_S.get())
        timer_time=TextValidator.validate_number(text=self.var_timer_time.get())
        if chessboard_dimensions is not None and square_size is not None and S is not None and timer_time is not None:
            if S >= 3 and square_size > 0 and timer_time > 0:
                return CalibrationParameters(chessboard_dimensions=chessboard_dimensions, square_size=square_size, S=S, timer_time=timer_time)
            else:
                return None
        else:
            return None