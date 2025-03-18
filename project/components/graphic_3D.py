import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

"""
Clase para graficar en 2D.

Se trabaja en metros (m).
"""
class Graphic3D():
    def __init__(self, square_size=2):
        self.fig=plt.figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1.2)
        self.ax=plt.axes(projection='3d')
        self.ax.view_init(elev=20., azim=-0, roll=0)
        self.square_size=square_size # En metros
        self.graphic_configuration(xlabel="X", ylabel="Y", zlabel="Z", xlim=[-self.square_size,self.square_size], ylim=[-self.square_size,self.square_size], zlim=[-self.square_size,self.square_size])

    # Para establecer las configuraciones del grafico principal
    def graphic_configuration(self, **kwargs):
        self.ax.set(**kwargs)

    # Para establecer el titulo del grafico principal
    def set_title(self, title):
        self.ax.set_title(title)

    # Para limpiar todos los dibujos realizados
    def clear(self):
        for line in self.ax.lines:
            line.remove()
        for collection in self.ax.collections:
            collection.remove()
        for text in self.ax.texts:
            text.remove()

    """
    Para dibujar un sistema de coordenadas.

    Se tiene una serie de sistemas de coordenadas (matrices de rotacion), donde cada uno se encuentra expresado con respecto
    al anterior.

    H contiene una matriz de rotacion y un vector de traslacion.

    Entrada: t_list ([(3,1), ...]), T_list ([(3,3), ...]), 
    Salida: H (4,4)
    """
    def plot_coordinate_system(self, t_list, T_list, length=1):
        H=None
        for t,T in list(zip(t_list, T_list)):
            H_=np.concatenate([np.concatenate([T,np.zeros((1,3))], axis=0), np.concatenate([t,np.ones((1,1))], axis=0)], axis=1)
            H=H_ if H is None else H @ H_
        vlast_=np.array([[0,0,0,1],[length,0,0,1],[0,length,0,1],[0,0,length,1]]).T 
        vm_=H @ vlast_
        # Puntos en forma de vector fila
        vm=vm_[0:3,:].T
        self.plot_lines(ps=vm[[0,1]], connection_list=[(0,1)], color_rgb=(255,0,0))
        self.plot_lines(ps=vm[[0,2]], connection_list=[(0,1)], color_rgb=(0,255,0))
        self.plot_lines(ps=vm[[0,3]], connection_list=[(0,1)], color_rgb=(0,0,255))
        self.plot_points(ps=vm, color_rgb=(0,0,0), alpha=0.8, marker=".", s=50)
        return H

    """
    Para dibujar puntos.

    Entrada: ps (m,3), color_rgb (tupla de 3 elementos)
    Salida: 
    """
    def plot_points(self, ps, color_rgb, alpha=0.8, marker=".", s=50):
        self.ax.scatter3D(ps[:,0], ps[:,1], ps[:,2], color=(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255), alpha=alpha, marker=marker, s=s)

    """
    Para dibujar lineas.

    Entrada: ps (m,3), connection_list ([(tupla de 2 elementos), ...]), color_rgb (tupla de 3 elementos)
    Salida: 
    """
    def plot_lines(self, ps, connection_list, color_rgb):
        for connection in connection_list:
            c1,c2=connection
            x1,y1,z1=ps[c1]
            x2,y2,z2=ps[c2]
            self.ax.plot3D([x1,x2], [y1,y2], [z1,z2], color=(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255))

    """
    Para dibujar un poligono.

    Entrada: verts (m,3)
    Salida: 
    """
    def plot_polygon(self, verts, alpha=0.8, facecolors="blue", edgecolors="black"):
        xs,ys,zs=verts[:,0],verts[:,1],verts[:,2]
        poly=Poly3DCollection([list(zip(xs,ys,zs))], alpha=alpha, facecolors=facecolors, edgecolors=edgecolors)
        self.ax.add_collection3d(poly)

    """
    Para dibujar textos.

    Entrada: ps (m,3), text_list ([(texto), ...]), color_rgb_list ([(tupla de 3 elementos), ...])
    Salida: 
    """
    def plot_texts(self, ps, text_list, color_rgb_list, fontsize="medium", fontfamily="sans-serif", fontweight="bold", horizontalalignment="center", verticalalignment="top"):
        for i in range(ps.shape[0]):
            x,y,z=ps[i]
            text=text_list[i]
            color_rgb=color_rgb_list[i]
            self.ax.text(x, y, z, s=text, color=(color_rgb[0] / 255, color_rgb[1] / 255, color_rgb[2] / 255), fontsize=fontsize, fontfamily=fontfamily, fontweight=fontweight, horizontalalignment=horizontalalignment, verticalalignment=verticalalignment)

    """
    Para dibujar un angulo (a partir de un poligono y un texto).

    Se requieren 3 puntos para poder definir un angulo entre ellos.

    pm: Punto medio
    p1: Un punto
    p2: Un punto

    Entrada: pm (3,), p1 (3,), p2 (3,)
    Salida: 
    """
    def plot_angle(self, pm, p1, p2, show_degree=True, color_rgb_polygon_facecolors=(0,255,0), color_rgb_polygon_edgecolors=(0,0,0), alpha_polygon=0.8, color_rgb_text=(0,0,0), fontsize_text="medium", fontweight_text="bold"):
        ac,ag=(p1 - pm, p2 - pm) if np.linalg.norm(p1 - pm) < np.linalg.norm(p2 - pm) else (p2 - pm,p1 - pm)
        n=100
        if np.linalg.norm(np.cross(ac,ag)) == 0:
            # Los vectores son paralelos (theta=0)
            verts=np.concatenate([pm[None,:], (pm + ac)[None,:]], axis=0)
            self.plot_polygon(verts=verts, alpha=alpha_polygon, facecolors=(color_rgb_polygon_facecolors[0] / 255, color_rgb_polygon_facecolors[1] / 255, color_rgb_polygon_facecolors[2] / 255), edgecolors=(color_rgb_polygon_edgecolors[0] / 255, color_rgb_polygon_edgecolors[1] / 255, color_rgb_polygon_edgecolors[2] / 255))
            if show_degree:
                ps_text=(pm + (1/2) * ac)[None,:]
                self.plot_texts(ps=ps_text, text_list=["0.00°"], color_rgb_list=[color_rgb_text], fontsize=fontsize_text, fontfamily="sans-serif", fontweight=fontweight_text, horizontalalignment="center", verticalalignment="center")
        else:
            # Los vectores no son paralelos
            norm_ac=np.linalg.norm(ac)
            norm_ag=np.linalg.norm(ag)
            theta=np.arccos(np.dot(ac,ag) / (norm_ac * norm_ag))
            b=ag - (np.dot(ac,ag) / norm_ac**2) * ac
            b=norm_ac * (b / np.linalg.norm(b))
            v=lambda alpha: pm + ((np.cos(alpha)) * ac + (np.sin(alpha)) * b)
            verts=np.zeros((n + 1,3))
            verts[0]=pm
            alpha_vals=np.linspace(0,theta,n) if np.linalg.norm(np.cross(v(alpha=theta) - pm, ag)) < np.linalg.norm(np.cross(v(alpha=-theta) - pm, ag)) else np.linspace(0,-theta,n)
            for i in range(n):
                alpha=alpha_vals[i]
                verts[i + 1]=v(alpha=alpha)
            self.plot_polygon(verts=verts, alpha=alpha_polygon, facecolors=(color_rgb_polygon_facecolors[0] / 255, color_rgb_polygon_facecolors[1] / 255, color_rgb_polygon_facecolors[2] / 255), edgecolors=(color_rgb_polygon_edgecolors[0] / 255, color_rgb_polygon_edgecolors[1] / 255, color_rgb_polygon_edgecolors[2] / 255))
            if show_degree:
                ps_text=(pm + (1/2) * (verts[verts.shape[0] // 2] - pm))[None,:]
                self.plot_texts(ps=ps_text, text_list=["{:,.2f}°".format(theta * (180 / np.pi))], color_rgb_list=[color_rgb_text], fontsize=fontsize_text, fontfamily="sans-serif", fontweight=fontweight_text, horizontalalignment="center", verticalalignment="center")







