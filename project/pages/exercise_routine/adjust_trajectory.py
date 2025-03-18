import numpy as np

class AdjustTrajectory():
    # Nombre del archivo de parametros de los modelos no lineales para cada punto 3D
    nonlinear_models_parameters_file_name="nonlinear_models_parameters.npy"

    # Para predecir puntos sobre la curva ajustada
    @staticmethod
    def predict(params, p0, n):
        # Parametros optimos
        t_min,t_max=params[0:2]
        wx1,wx2,wy1,wy2,wz1,wz2=params[2:]
        # Parametrizacion de la curva ajustada
        p=lambda t: p0 + np.array([wx1 * t + wx2 * t ** 2, wy1 * t + wy2 * t ** 2, wz1 * t + wz2 * t ** 2])
        # Puntos 3D sobre la curva ajustada
        point_3D_over_time=np.concatenate([p(t)[None,:] for t in np.linspace(t_min, t_max, n)], axis=0)
        return point_3D_over_time

    # Para ajustar posturas 3D y trayectorias de los puntos 3D a lo largo del tiempo
    @staticmethod
    def get_adjusted_posture_3D_and_point_3D_over_time_list(posture_3D_over_time_list, nonlinear_models_parameters, algorithm_number_points, n=100):
        # n: Numero de puntos que se obtendran a lo largo de la trayectoria ajustada
        adjusted_posture_3D_over_time_list=[np.zeros((algorithm_number_points,3)) for i in range(n)]
        adjusted_point_3D_over_time_list=[]
        # Para cada punto 3D
        for i in range(algorithm_number_points):
            # Trayectoria ajustada de un solo punto 
            adjusted_point_3D_over_time=AdjustTrajectory.predict(params=nonlinear_models_parameters[i], p0=posture_3D_over_time_list[0][i], n=n)
            adjusted_point_3D_over_time_list.append(adjusted_point_3D_over_time)
            for j in range(n):
                # Punto 3D en un instante de tiempo para un punto 3D especifico
                adjusted_posture_3D_over_time_list[j][i]=adjusted_point_3D_over_time[j]
        return adjusted_posture_3D_over_time_list,adjusted_point_3D_over_time_list