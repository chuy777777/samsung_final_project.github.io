import cv2
import numpy as np

dict_aruco = dict({
	"DICT_4X4_50" : cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100" : cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250" : cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000" : cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50" : cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100" : cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250" : cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000" : cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50" : cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100" : cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250" : cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000" : cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50" : cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100" : cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250" : cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000" : cv2.aruco.DICT_7X7_1000,
})

"""
Clase para obtener puntos 2D y 3D de patrones conocidos:
    - Marcador ArUco
    - Tablero de ajedrez

Los puntos 2D son detectados por algoritmos de OpenCV.

Los puntos 3D estan calculados a partir de como esta definido el sistema de coordenadas del mundo sobre el patron conocido.
"""
class ImagePoints2D3D():
    def __init__(self):
        pass
    
    # Devuelve puntos 2D (distorsionados) sobre un marcador ArUco
    @staticmethod
    def get_v3pds_from_aruco_marker_image(frame_bgr, aruco_type, aruco_id):
        if frame_bgr is not None:
            aruco_dict=cv2.aruco.Dictionary_get(dict_aruco[aruco_type])
            aruco_params=cv2.aruco.DetectorParameters_create()
            # Las detecciones se hacen en sentido de las manecillas del reloj
            corners, ids, rejected=cv2.aruco.detectMarkers(frame_bgr, aruco_dict, parameters=aruco_params) 
            if len(corners) > 0:
                res=np.where(ids.squeeze() == aruco_id)[0]
                if len(res) > 0:
                    pos=np.where(ids.squeeze() == aruco_id)[0][0]
                    v3pds=corners[pos].squeeze()
                    return v3pds
                
        return None

    # Devuelve puntos 3D sobre un marcador ArUco
    @staticmethod
    def get_vws_from_aruco_marker(square_size):
        T=np.eye(2) * square_size
        vws=T @ np.array([[0,1,1,0],[0,0,1,1]])
        vws=np.concatenate([vws, np.zeros((1, vws.shape[1]))], axis=0).T

        return vws
    
    # Devuelve puntos 2D (distorsionados) sobre un tablero de ajedrez
    @staticmethod
    def get_v3pds_from_chessboard_image(frame_bgr, chessboard_dimensions, fast=True):
        if frame_bgr is not None:
            frame_gray=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # Las detecciones se hacen de izquierda a derecha, de arriba a abajo
            ret,corners=cv2.findChessboardCorners(frame_gray, chessboard_dimensions, None, flags=cv2.CALIB_CB_FAST_CHECK) if fast else cv2.findChessboardCorners(frame_gray, chessboard_dimensions, None)
            if ret:
                v3pds=corners.squeeze()
                return v3pds
            
        return None

    # Devuelve puntos 3D sobre un tablero de ajedrez
    @staticmethod
    def get_vws_from_chessboard(chessboard_dimensions, square_size):
        N=chessboard_dimensions[0] * chessboard_dimensions[1]
        vws=np.zeros((N,3))
        count=0
        T=np.eye(2) * square_size
        for y in range(chessboard_dimensions[1]):
            for x in range(chessboard_dimensions[0]):
                v=np.array([[x],[y]])
                u=(T @ v).flatten()
                vws[count]=np.array([u[0],u[1],0])
                count+=1

        return vws

    # Devuelve pares de puntos correspondientes 2D (distorsionados) y 3D sobre un marcador ArUco
    @staticmethod
    def point_pairs_2D_and_3D_aruco_marker(frame_bgr, aruco_type, aruco_id, square_size):
        # v3pds (4,2), vws (4,2)
        v3pds=ImagePoints2D3D.get_v3pds_from_aruco_marker_image(frame_bgr=frame_bgr, aruco_type=aruco_type, aruco_id=aruco_id)
        vws=ImagePoints2D3D.get_vws_from_aruco_marker(square_size=square_size)
       
        return (v3pds,vws,True) if v3pds is not None and vws is not None else (None,None,False)

    # Devuelve pares de puntos correspondientes 2D (distorsionados) y 3D sobre un tablero de ajedrez
    @staticmethod
    def point_pairs_2D_and_3D_chessboard(frame_bgr, chessboard_dimensions, square_size, fast=True):
        # v3pds (N,2), vws (N,2)
        v3pds=ImagePoints2D3D.get_v3pds_from_chessboard_image(frame_bgr=frame_bgr, chessboard_dimensions=chessboard_dimensions, fast=fast)
        vws=ImagePoints2D3D.get_vws_from_chessboard(chessboard_dimensions=chessboard_dimensions, square_size=square_size)
        
        return (v3pds,vws,True) if v3pds is not None and vws is not None else (None,None,False)