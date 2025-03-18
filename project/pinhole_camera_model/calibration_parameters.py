import pickle
import os 

"""
Clase para guardar los parametros de calibracion.

Recordar que el proceso de calibracion consta de tomar varias imagenes de un tablero de ajedrez en
diferentes posiciones y orientaciones.

Parametros:
    - chessboard_dimensions
        Dimensiones del tablero.
        Numero de esquinas interiores, tanto en ancho como en alto (width,height).
    - square_size
        Tamaño de un cuadrado del tablero de ajedrez (en milimetros).
    - S
        Numero de imagenes del tablero de ajedrez.
    - timer_time
        Tiempo entre cada imagen del tablero de ajedrez (en segundos).
"""
class CalibrationParameters():
    def __init__(self, chessboard_dimensions, square_size, S, timer_time):
        self.chessboard_dimensions=chessboard_dimensions
        self.square_size=square_size
        self.S=S
        self.timer_time=timer_time

    # Para guardar los parametros de calibracion en la computadora
    def save(self, full_path):
        with open(os.path.join(full_path, *["chessboard_dimensions.pickle"]), 'wb') as file:
            pickle.dump(self.chessboard_dimensions, file)
        with open(os.path.join(full_path, *["square_size.pickle"]), 'wb') as file:
            pickle.dump(self.square_size, file)
        with open(os.path.join(full_path, *["S.pickle"]), 'wb') as file:
            pickle.dump(self.S, file)
        with open(os.path.join(full_path, *["timer_time.pickle"]), 'wb') as file:
            pickle.dump(self.timer_time, file)

    # Para cargar los parametros de calibracion 
    @staticmethod
    def load(full_path):
        chessboard_dimensions=None 
        square_size=None
        S=None
        timer_time=None
        with open(os.path.join(full_path, *["chessboard_dimensions.pickle"]), 'rb') as file:
            chessboard_dimensions=pickle.load(file)
        with open(os.path.join(full_path, *["square_size.pickle"]), 'rb') as file:
            square_size=pickle.load(file)
        with open(os.path.join(full_path, *["S.pickle"]), 'rb') as file:
            S=pickle.load(file)
        with open(os.path.join(full_path, *["timer_time.pickle"]), 'rb') as file:
            timer_time=pickle.load(file)
        return CalibrationParameters(chessboard_dimensions=chessboard_dimensions, square_size=square_size, S=S, timer_time=timer_time)
    
    # Para imprimir los parametros de calibracion
    def __str__(self):
        return "Chessboard dimensions: {}\nSquare size: {}\nS: {}\nTimer time: {}".format(self.chessboard_dimensions, self.square_size, self.S, self.timer_time)