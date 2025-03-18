from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.graphic_2D import Graphic2D

"""
Frame para mostrar graficos en 2D.
"""
class FrameGraphic2D(CreateFrame, Graphic2D):
    def __init__(self, master, name, width=400, height=400, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(1,1), arr=None), **kwargs)
        Graphic2D.__init__(self)

        self.canvas=FigureCanvasTkAgg(figure=self.fig, master=self)
        self.canvas.get_tk_widget().config(width=width, height=height)

        self.insert_element(cad_pos="0,0", element=self.canvas.get_tk_widget(), padx=5, pady=5, sticky="")

    # Para actualizar el grafico.
    def draw(self):
        self.canvas.draw()