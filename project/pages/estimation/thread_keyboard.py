from threading import Thread
from pynput import keyboard

"""
Clase que ejecuta un hilo (thread) para la deteccion del pulsado de teclas.
"""
class ThreadKeyboard(Thread):
    def __init__(self, callback):
        Thread.__init__(self, daemon=True)
        self.callback=callback

    # Para iniciar el hilo
    def run(self):
        self.loop_keyboard()
        print("End ThreadKeyboard")    

    # Para detectar cuando una tecla se ha presionado
    def on_press(self, key):
        self.callback(key)

    # Para detectar cuando una tecla se ha soltado
    def on_release(self, key):
        # print(f"'{key}' released")
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Funcion que ejecuta el hilo
    def loop_keyboard(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
            listener.join()