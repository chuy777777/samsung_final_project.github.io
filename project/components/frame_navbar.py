from components.create_frame import CreateFrame
from components.grid_frame import GridFrame
from components.frame_page_buttons import FramePageButtons
from components.frame_select_camera import FrameSelectCamera

"""
Frame para mostrar los componentes de navegacion (para navegar entre componentes principales).
"""
class FrameNavbar(CreateFrame):
    def __init__(self, master, name, **kwargs):
        CreateFrame.__init__(self, master=master, name=name, grid_frame=GridFrame(dim=(2,1), arr=None), **kwargs)

        frame_button_pages=FramePageButtons(self, name="FramePageButtons")
        frame_select_camera=FrameSelectCamera(self, name="FrameSelectCamera")

        self.insert_element(cad_pos="0,0", element=frame_button_pages, padx=5, pady=5, sticky="")
        self.insert_element(cad_pos="1,0", element=frame_select_camera, padx=5, pady=5, sticky="")
