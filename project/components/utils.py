import numpy as np
import cv2
import customtkinter  as ctk
from PIL import Image
import os

# Para escalar un frame de la camara
def resize_frame_bgr(scale_percent, frame_bgr):
    if frame_bgr is not None:
        width = int(frame_bgr.shape[1] * scale_percent / 100)
        height = int(frame_bgr.shape[0] * scale_percent / 100)
        dim = (width, height)
        new_frame_bgr = cv2.resize(frame_bgr, dim, interpolation = cv2.INTER_AREA)
        return new_frame_bgr
    return None

# Para convertir un frame de la camara en una imagen de CustomTkinter
def frame_bgr_to_ctk_img(frame_bgr):
    if frame_bgr is not None:
        frame_rgb=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img=Image.fromarray(frame_rgb)
        img=ctk.CTkImage(img,size=(frame_bgr.shape[1],frame_bgr.shape[0]))
        return img
    return None

# Para convertir una imagen en una imagen de CustomTkinter
def image_to_ctk_img(image_full_path, width, height):
    if os.path.exists(image_full_path):
        return ctk.CTkImage(Image.open(image_full_path), size=(width, height))
    else:
        return None

# Para convertir un color RGB a un color HEX
def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))

# Para normalizar un vector
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

# Para generar un numero aleatorio en el rango [a,b]
def random_value_in_range(a,b):
    return a + np.random.random() * (b - a)

"""
Para generar una matriz de rotacion alrededor del eje 'X'.

El sistema de coordenadas que se utiliza es el tradicional (matplotlib).

Angulo en radianes.
"""
def Rx(psi):
    R=np.array([[1,0,0],[0,np.cos(psi),-np.sin(psi)],[0,np.sin(psi),np.cos(psi)]])
    return R

"""
Para generar una matriz de rotacion alrededor del eje 'Y'.

El sistema de coordenadas que se utiliza es el tradicional (matplotlib).

Angulo en radianes.
"""
def Ry(theta):
    R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    return R

"""
Para generar una matriz de rotacion alrededor del eje 'Z'.

El sistema de coordenadas que se utiliza es el tradicional (matplotlib).

Angulo en radianes.
"""
def Rz(phi):
    R=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    return R

"""
Para generar una matriz de rotacion completa en 3D.

El sistema de coordenadas que se utiliza es el tradicional (matplotlib).

Angulo en radianes.
"""
def RzRyRx(psi, theta, phi):
    R=Rz(phi=phi)@Ry(theta=theta)@Rx(psi=psi)
    return R

"""
Para obtener los angulos de Euler (3 angulos de la matriz de rotacion) a partir de una matriz de rotacion.

El sistema de coordenadas que se utiliza es el tradicional (matplotlib).

Matriz de rotacion de referencia:
    R=RzRyRx
"""
def euler_angles_from_to_rotation_matrix(R):
    R11,R12,R13,R21,R22,R23,R31,R32,R33=R.flatten(order='C')
    euler_angles=np.zeros(3)
    if R31 != -1 and R31 != 1:
        theta_1=np.arcsin(-R31)
        theta_2=np.pi - theta_1
        psi_1=np.arctan2(R32 / np.cos(theta_1), R33 / np.cos(theta_1))
        psi_2=np.arctan2(R32 / np.cos(theta_2), R33 / np.cos(theta_2))
        phi_1=np.arctan2(R21 / np.cos(theta_1), R11 / np.cos(theta_1))
        phi_2=np.arctan2(R21 / np.cos(theta_2), R11 / np.cos(theta_2))
        euler_angles=np.array([psi_1, theta_1, phi_1])
    else:
        if R31 == -1:
            theta=np.pi/2
            phi=0  
            psi=np.arctan2(R12, R13) + phi
            euler_angles=np.array([psi, theta, phi])
        else:
            theta=-np.pi/2
            phi=0  
            psi=np.arctan2(-R12, -R13) - phi
            euler_angles=np.array([psi, theta, phi])
    return euler_angles