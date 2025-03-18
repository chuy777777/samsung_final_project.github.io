import numpy as np
import cv2

"""
Clase para dibujar sobre una imagen.
"""
class DrawImage():
    def __init__(self):
        pass

    """
    Dibuja puntos sobre una imagen.
    
    Entrada: frame_bgr (frame de la camara), points (N,2)
    Salida: frame_bgr_copy (frame de la camara con dibujo)
    """
    @staticmethod
    def draw_points(frame_bgr, points, size=5, color_bgr=(0,255,0)):
        frame_bgr_copy=np.copy(frame_bgr)
        N=points.shape[0]
        for n in range(N):
            x,y=points[n].astype(int)
            frame_bgr_copy=cv2.circle(frame_bgr_copy, (x,y), size, color_bgr, -1)

        return frame_bgr_copy

    """
    Dibuja lineas sobre una imagen.
    
    Entrada: frame_bgr (frame de la camara), lines ((N,2), (N,2))
    Salida: frame_bgr_copy (frame de la camara con dibujo)
    """
    @staticmethod
    def draw_lines(frame_bgr, lines, color_bgr=(0,0,255), weight=2):
        frame_bgr_copy=np.copy(frame_bgr)
        points1,points2=lines
        N=points1.shape[0]
        for n in range(N):
            x1,y1=points1[n].astype(int)
            x2,y2=points2[n].astype(int)
            frame_bgr_copy=cv2.line(frame_bgr_copy, (x1,y1), (x2,y2), color_bgr, weight)

        return frame_bgr_copy

    """
    Dibuja conexiones sobre una imagen.
    
    Entrada: frame_bgr (frame de la camara), points (N,2), connection_list ([(tupla de 2 elementos), ...])
    Salida: frame_bgr_copy (frame de la camara con dibujo)
    """
    @staticmethod
    def draw_conenctions(frame_bgr, points, connection_list):
        N=len(connection_list)
        points1,points2=np.zeros((N,2)),np.zeros((N,2))
        for n in range(N):
            c1,c2=connection_list[n]
            points1[n]=points[c1]
            points2[n]=points[c2]
        frame_bgr=DrawImage.draw_points(frame_bgr=frame_bgr, points=points, size=5, color_bgr=(0,0,255))
        frame_bgr=DrawImage.draw_lines(frame_bgr=frame_bgr, lines=(points1,points2), color_bgr=(0,255,0), weight=2)
        return frame_bgr



               