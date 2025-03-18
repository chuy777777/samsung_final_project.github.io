from threading import Event

"""
Clase plantilla para los hilos (threads) de las camaras.
"""
class TemplateThread():
    def __init__(self):
        self.event_kill_thread=Event() 

    def kill_thread(self):
        self.event_kill_thread.set()


