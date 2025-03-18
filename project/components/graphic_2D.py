import matplotlib.pyplot as plt

"""
Clase para graficar en 2D.
"""
class Graphic2D():
    def __init__(self):
        self.fig=plt.figure()
        self.ax=plt.subplot()
        self.ax.grid()

    # Para establecer las configuraciones del grafico principal
    def graphic_configuration(self, **kwargs):
        self.ax.set(**kwargs)

    # Para limpiar todos los dibujos realizados
    def clear(self):
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()

    """
    Para dibujar lineas.

    Entrada: ps (m,2), color_rgb (tupla de 3 elementos)
    Salida: 
    """
    def plot_lines(self, ps, color_rgb):
        self.ax.plot(ps[:,0], ps[:,1], color=(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255))

    """
    Para dibujar puntos.

    Entrada: ps (m,2), color_rgb (tupla de 3 elementos)
    Salida: 
    """
    def plot_points(self, ps, color_rgb, marker=".", s=50):
        self.ax.scatter(ps[:,0], ps[:,1], color=(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255), marker=marker, s=s)








