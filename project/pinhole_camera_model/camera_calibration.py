import numpy as np
from pinhole_camera_model.distorted_to_undistorted import DistortedToUndistorted
from framework_ml.optimizers import Adam

"""
Clase que contiene todas las funciones necesarias para el proceso de calibracion de la camara.
"""
class CameraCalibration():
    def __init__(self):
        pass
    
    # Toma un vector columna y lo devuelve a su forma original de dimensiones "shape"
    @staticmethod
    def inv_vec(x, shape):
        if len(shape) == 0:   
            return x.squeeze()
        elif len(shape) == 1:   
            w=shape   
            return x.flatten()
        elif len(shape) == 2:   
            h,w=shape   
            return np.reshape(x, shape, order='F')
        elif len(shape) == 3:
            d,h,w=shape
            return np.transpose(np.reshape(x, (d,w,h), order='C'), axes=(0,2,1))
        elif len(shape) == 4:
            s,d,h,w=shape
            return np.transpose(np.reshape(x, (s,d,w,h), order='C'), axes=(0,1,3,2))

    # Devuelve un vector columna de dimensiones (p,1)
    @staticmethod
    def vec(X):
        if len(X.shape) == 0:   
            return np.expand_dims(X, axis=(0,1))
        elif len(X.shape) == 1:   
            return X[:,None]
        elif len(X.shape) == 2:      
            return X.flatten(order='F')[:,None]
        elif len(X.shape) == 3:
            return np.transpose(X, axes=(0,2,1)).flatten(order='C')[:,None]
        elif len(X.shape) == 4:
            return np.transpose(X, axes=(0,1,3,2)).flatten(order='C')[:,None]

    """
    Imprime informacion con respecto a la matriz de rotacion que viene incluida en la matriz extrinseca de la camara.

    Se debe cumplir lo siguiente:
        - Los vectores base de la matriz de rotacion deben formar un conjunto de vectores ortonormales
            - Deben tener norma 1
            - Deben ser ortogonales entre si

    Entrada: Qs (S,3,4)
    Salida: 
    """
    @staticmethod
    def print_info_Twc(Qs):
        S=Qs.shape[0]
        for s in range(S):
            Q=Qs[s]
            print("iwc*iwc (1): {:.5f} jwc*jwc (1): {:.5f} kwc*kwc (1): {:.5f} iwc*jwc (0): {:.5f} iwc*kwc (0): {:.5f} jwc*kwc (0): {:.5f}".format(np.dot(Q[:,0],Q[:,0]), np.dot(Q[:,1],Q[:,1]), np.dot(Q[:,2],Q[:,2]), np.dot(Q[:,0],Q[:,1]), np.dot(Q[:,0],Q[:,2]), np.dot(Q[:,1],Q[:,2])))

    """
    Devuelve un vector de parametros para la funcion objetivo (funcion lagrangiana que se optimiza en el proceso de calibracion de la camara).

    Entrada: K (3,3), q (5,1), Q_s (S,3,3), lambdas (3 * S,1)
    Salida: x (5 + 5 + S * 3 * 3 + 3 * S,)
    """
    @staticmethod
    def get_x(K, q, Q_s, lambdas):
        k1,k2,k3,k4,k5=K[0,0],K[0,1],K[1,1],K[0,2],K[1,2]
        x=np.concatenate([CameraCalibration.vec(X=np.array([k1,k2,k3,k4,k5])[:,None]), CameraCalibration.vec(X=q), CameraCalibration.vec(X=Q_s), CameraCalibration.vec(X=lambdas)], axis=0).flatten()
        
        return x

    """
    Devuelve todos los parametros a partir del vector de parametros.

    Entrada: x (5 + 5 + S * 3 * 3 + 3 * S,), S (un escalar)
    Salida: K (3,3), q (5,1), Q_s (S,3,3), lambdas (3 * S,1), parameter_slices (para cada parametro, el 'slice' correspondiente)
    """
    @staticmethod
    def get_parameters(x, S):
        q_shape=(5,1)
        Q_s_shape=(S,3,3)
        lambdas_shape=(3 * S,1)

        accum=0
        K_slice=slice(0,5)
        k1,k2,k3,k4,k5=x[K_slice]
        K=np.array([[k1,k2,k4],[0,k3,k5],[0,0,1]])
        accum+=5
        q_slice=slice(accum,accum + np.prod(q_shape))
        q=CameraCalibration.inv_vec(x=x[q_slice][:,None], shape=q_shape)
        accum+=np.prod(q_shape)
        Q_s_slice=slice(accum,accum + np.prod(Q_s_shape))
        Q_s_slice_list=[]
        for s in range(S):
            Q_s_slice_list.append(slice(accum + 9 * s,accum + 9 * (s + 1)))
        Q_s=CameraCalibration.inv_vec(x=x[Q_s_slice][:,None], shape=Q_s_shape)
        accum+=np.prod(Q_s_shape)
        lambdas_slice=slice(accum,accum + np.prod(lambdas_shape))
        lambdas_slice_list=[]
        for s in range(S):
            lambdas_slice_list.append(slice(accum + 3 * s,accum + 3 * (s + 1)))
        lambdas=CameraCalibration.inv_vec(x=x[lambdas_slice][:,None], shape=lambdas_shape)

        parameter_slices=(K_slice,q_slice,Q_s_slice_list,lambdas_slice_list)

        return K,q,Q_s,lambdas,parameter_slices
    
    """
    Devuelve la matriz extrinseca Q a partir de la matriz extrinseca Q simplificada (no cuenta con un vector columna).

    Entrada: Q_s (S,3,3)
    Salida: Qs (S,3,4)
    """
    @staticmethod
    def get_Qs_from_Q_s(Q_s):
        S=Q_s.shape[0]
        Qs=np.zeros((S,3,4))
        for s in range(S):
            iwc,jwc,twc=Q_s[s,:,0][:,None],Q_s[s,:,1][:,None],Q_s[s,:,2][:,None]
            # Se sigue la regla de la mano derecha (para que coincida con el eje Z del sistema de coordenadas del mundo)
            kwc=-np.cross(iwc.flatten(),jwc.flatten())[:,None]
            Q=np.concatenate([iwc,jwc,kwc,twc], axis=1)
            Qs[s]=Q

        return Qs

    """
    Devuelve matrices G a partir de pares de puntos correspondientes 2D (sin distorsion) y 3D.

    La matriz G resulta de manipular el modelo de camara Pinhole ideal, y haciendo que toda componente Z del punto 3D sea igual a 0
    (es decir, todo punto 3D vive sobre un plano, el plano del tablero de ajedrez).

    La matriz G ayuda a generar algunas ecuaciones para el calculo de la matriz B (por tanto, se necesitan varias matrices G).

    Entrada: v3ps_list ([(N,2), ...]), vws (N,3)
    Salida: Gs (S,3,3)
    """
    @staticmethod
    def get_Gs(v3ps_list, vws):
        S=len(v3ps_list)
        Gs=np.zeros((S,3,3))
        for s in range(S):
            v3ps=v3ps_list[s]
            N=v3ps.shape[0]
            A=np.zeros((2 * N,8))
            b=np.zeros((2 * N,1))
            for n in range(N):
                v3px,v3py=v3ps[n]
                vwx,vwy,vwz=vws[n]
                A_=np.array([[vwx, vwy, 1, 0, 0, 0, -vwx * v3px, -vwy * v3px],[0, 0, 0, vwx, vwy, 1, -vwx * v3py, -vwy * v3py]])
                b_=np.array([[v3px],[v3py]])
                A[2 * n:2 * (n + 1)]=A_
                b[2 * n:2 * (n + 1)]=b_
            x=np.linalg.inv(A.T @ A) @ A.T @ b 
            x=np.concatenate([x, np.ones((1,1))], axis=0)
            G=np.reshape(x, (3,3), order='C')
            Gs[s]=G

        return Gs

    """
    Devuelve la matriz B.

    La matriz B se calcula a partir de varias matrices G.

    A partir de la matriz B es posible calcular la matriz intrinseca K por medio de una descomposicion de Cholesky sobre B.

    Entrada: Gs (S,3,3)
    Salida: B (3,3)
    """
    @staticmethod
    def get_B(Gs):
        # Descomposicion SVD de una matriz
        # u,s,vh=np.linalg.svd(A)
        # The rows of vh are the eigenvectors of A'A and the columns of u are the eigenvectors of AA'. 
        # In both cases the corresponding (possibly non-zero) eigenvalues are given by s**2.
        S=Gs.shape[0]
        A=np.zeros((2 * S,6))
        for s in range(S):
            G=Gs[s]
            g11,g12,g13,g21,g22,g23,g31,g32,_=G.flatten(order='C')
            A_=np.array([[g11**2 - g12**2, 2 * g11 * g21 - 2 * g12 * g22, 2 * g11 * g31 - 2 * g12 * g32, g21**2 - g22**2, 2 * g21 * g31 - 2 * g22 * g32, g31**2 - g32**2],[g11 * g12, g12 * g21 + g11 * g22, g12 * g31 + g11 * g32, g21 * g22, g22 * g31 + g21 * g32, g31 * g32]])
            A[2 * s:2 * (s + 1),:]=A_
        u,s,vh=np.linalg.svd(A)
        h=np.reshape(vh[-1],(6,1))
        a,b,c,d,e,f=h.flatten()
        B=np.array([[a,b,c],[b,d,e],[c,e,f]])

        return B

    """
    Devuelve la matriz intrinseca K y las matrices extrinsecas Q.

    Entrada: B (3,3), Gs (S,3,3)
    Salida: K (3,3), Qs (S,3,4)
    """
    @staticmethod
    def get_K_Qs(B, Gs):
        L=np.linalg.cholesky(B)
        K=np.linalg.inv(L.T)
        K_inv=L.T
        S=Gs.shape[0]
        Qs=np.zeros((S,3,4))
        for s in range(S):
            G=Gs[s]
            g1,g2,g3=G[:,0][:,None],G[:,1][:,None],G[:,2][:,None]
            h33=1 / np.linalg.norm(K_inv @ g1)
            iwc=h33 * K_inv @ g1
            jwc=h33 * K_inv @ g2
            kwc=-np.cross(iwc.flatten(),jwc.flatten())[:,None]
            twc=h33 * K_inv @ g3
            Q=np.concatenate([iwc,jwc,kwc,twc], axis=1)
            Qs[s]=Q

        return K,Qs

    """
    Devuelve la matriz extrinseca Q.
    
    Entrada: v3pds (N,2), vws (N,3), K (3,3), K_inv (3,3), q (5,1)
    Salida: Q (3,4)
    """
    @staticmethod
    def get_Q(v3pds, vws, K, K_inv, q):
        v3ps=DistortedToUndistorted.undistorted_points(v3pds=v3pds, K=K, K_inv=K_inv, q=q)
        G=CameraCalibration.get_Gs(v3ps_list=[v3ps], vws=vws)[0]
        g1,g2,g3=G[:,0][:,None],G[:,1][:,None],G[:,2][:,None]
        h33=1 / np.linalg.norm(K_inv @ g1)
        iwc=h33 * K_inv @ g1
        jwc=h33 * K_inv @ g2
        kwc=-np.cross(iwc.flatten(),jwc.flatten())[:,None]
        twc=h33 * K_inv @ g3
        Q=np.concatenate([iwc,jwc,kwc,twc], axis=1)

        return Q
    
    """
    Devuelve los puntos 2D (distorsionados) sobre la imagen.
    
    Entrada: vws (N,3), K (3,3), Q (3,4), q (5,1)
    Salida: v3pds (N,2)
    """
    @staticmethod
    def get_v3pds(vws, K, Q, q):
        N=vws.shape[0]
        k1,k2,k3,p1,p2=q.flatten()
        v3pds=np.zeros((N,2))
        for n in range(N):
            vw=vws[n][:,None]
            M=np.concatenate([vw, np.ones((1,1))], axis=0)
            vc=Q @ M
            vcx,vcy,vcz=vc.flatten()
            v1n=np.array([[vcx / vcz], [vcy / vcz]])
            v1nx,v1ny=v1n.flatten()
            r=np.linalg.norm(v1n)
            d=np.array([v1nx * (k1 * r**2 + k2 * r**4 + k3 * r**6) + 2 * p1 * v1nx * v1ny + p2 * (r**2 + 2 * v1nx**2),v1ny * (k1 * r**2 + k2 * r**4 + k3 * r**6) + p1 * (r**2 + 2 * v1ny**2)+ 2 * p2 * v1nx * v1ny ])[:,None]
            v1nd=v1n + d 
            Mnd=np.concatenate([v1nd, np.ones((1,1))], axis=0)
            md=K @ Mnd
            v3pd=md[0:2]
            v3pds[n]=v3pd.flatten()

        return v3pds
    
    """
    Devuelve una buena inicializacion de los parametros para el proceso de optimizacion.
    
    Entrada: v3pds_list ([(N,2), ...]), vws (N,3)
    Salida: K (3,3), q (5,1), Q_s (S,3,3), lambdas (3 * S,1)
    """
    @staticmethod
    def initialization_of_parameters(v3pds_list, vws):
        # Se asume que los puntos no estan distorsionados (que pertenecen al modelo ideal)
        v3ps_list=v3pds_list
        S=len(v3ps_list)
        Gs=CameraCalibration.get_Gs(v3ps_list=v3ps_list, vws=vws)
        B=CameraCalibration.get_B(Gs=Gs)
        K,Qs=CameraCalibration.get_K_Qs(B=B, Gs=Gs)
        # Hacer lo siguiente nos da una mejor inicializacion de K 
        K/=K[2,2] 
        Q_s=Qs[:,:,[0,1,3]]
        q=np.zeros((5,1))
        lambdas=np.zeros((3 * S,1))

        return K,q,Q_s,lambdas

    """
    Funcion para el proceso de optimizacion (funcion lagrangiana).
    
    Entrada: x (5 + 5 + S * 3 * 3 + 3 * S,), v3pds_list ([(N,2), ...]), vws (N,3), S (un escalar)
    Salida: L (un escalar), grad (5 + 5 + S * 3 * 3 + 3 * S,)
    """
    @staticmethod
    def loss_function(x, v3pds_list, vws, S):
        S,N=len(v3pds_list),v3pds_list[0].shape[0]
        K,q,Q_s,lambdas,parameter_slices=CameraCalibration.get_parameters(x=x, S=S)
        K_slice,q_slice,Q_s_slice_list,lambdas_slice_list=parameter_slices
        
        L=0
        grad=np.zeros(x.shape)
        for s in range(S):
            v3pds=v3pds_list[s]
            Q_=Q_s[s]
            for n in range(N):
                v3pdr=v3pds[n][:,None]
                vwx,vwy,vwz=vws[n]
                M_=np.array([[vwx],[vwy],[1]])
                vc=Q_ @ M_
                vcx,vcy,vcz=vc.flatten()
                v1n=np.array([[vcx / vcz],[vcy / vcz]])
                v1nx,v1ny=v1n.flatten()
                r=np.linalg.norm(v1n)
                A=np.array([[v1nx * r**2, v1nx * r**4, v1nx * r**6, 2 * v1nx * v1ny, r**2 + 2 * v1nx**2],[v1ny * r**2, v1ny * r**4, v1ny * r**6, r**2 + 2 * v1ny**2, 2 * v1nx * v1ny]])
                d=A @ q
                v1nd=v1n + d
                And=np.array([[1,0],[0,1],[0,0]])
                bnd=np.array([[0],[0],[1]])
                Mnd=And @ v1nd + bnd
                Mndx,Mndy,Mndz=Mnd.flatten()
                md=K @ Mnd
                mdr=np.concatenate([v3pdr, np.ones((1,1))], axis=0)
                e=md - mdr 
                L+=np.linalg.norm(e) ** 2

                dgde=2 * e.T
                dedmd=np.eye(3)
                dmddk=np.array([[Mndx,Mndy,0,Mndz,0],[0,0,Mndy,0,Mndz],[0,0,0,0,0]])
                dmddMnd=K 
                dMnddv1nd=And 
                dv1nddd=np.eye(2)
                dddq=A
                dv1nddv1n=np.eye(2) + (np.kron(q.T,np.eye(2)) @ np.array([[r**2 + 2 * v1nx**2,2 * v1nx * v1ny],[2 * v1nx * v1ny,r**2 + 2 * v1ny**2],[r**4 + 4 * r**2 * v1nx**2,4 * r**2 * v1nx * v1ny],[4 * r**2 * v1nx * v1ny,r**4 + 4 * r**2 * v1ny**2],[r**6 + 6 * r**4 * v1nx**2,6 * r**4 * v1nx * v1ny],[6 * r**4 * v1nx * v1ny,r**6 + 6 * r**4 * v1ny**2],[2 * v1ny,2 * v1nx],[2 * v1nx,6 * v1ny],[6 * v1nx,2 * v1ny],[2 * v1ny,2 * v1nx]]))
                dv1ndvc=np.array([[1 / vcz,0,-(vcx / vcz**2)],[0,1 / vcz,-(vcy / vcz**2)]])
                dvcdt=np.kron(M_.T,np.eye(3))

                dgdk=dgde @ dedmd @ dmddk
                dgdq=dgde @ dedmd @ dmddMnd @ dMnddv1nd @ dv1nddd @ dddq
                dgdt=dgde @ dedmd @ dmddMnd @ dMnddv1nd @ dv1nddv1n @ dv1ndvc @ dvcdt

                grad[K_slice]+=dgdk.flatten()
                grad[q_slice]+=dgdq.flatten()
                grad[Q_s_slice_list[s]]+=dgdt.flatten()
                
        for s in range(S):
            lambda1,lambda2,lambda3=lambdas[3 * s:3 * (s+1),0]
            Q_=Q_s[s]
            iwc,jwc=Q_[:,0],Q_[:,1]
            iwcx,iwcy,iwcz=iwc
            jwcx,jwcy,jwcz=jwc

            L-=(lambda1 * (np.dot(iwc,iwc) - 1) + lambda2 * (np.dot(jwc,jwc) - 1) + lambda3 * np.dot(iwc,jwc))

            grad[Q_s_slice_list[s]]-=np.array([2 * iwcx * lambda1 + jwcx * lambda3,2 * iwcy * lambda1 + jwcy * lambda3,2 * iwcz * lambda1 + jwcz * lambda3,2 * jwcx * lambda2 + iwcx * lambda3,2 * jwcy * lambda2 + iwcy * lambda3,2 * jwcz * lambda2 + iwcz * lambda3,0,0,0])
            grad[lambdas_slice_list[s]]-=np.array([np.dot(iwc,iwc) - 1,np.dot(jwc,jwc) - 1,np.dot(iwc,jwc)])
        
        # Costo de la funcion y gradiente de la funcion
        return L,grad

    """
    Para calcular los parametros mas optimos de la camara (proceso de calibracion).
    
    Entrada: v3pds_list ([(N,2), ...]), vws (N,3)
    Salida: K_opt (3,3), q_opt (5,1), Qs_opt (S,3,4), lambdas_opt (3 * S,1), history_L (num_it,), history_norm_grad (num_it,), is_ok (booleano)
    """
    @staticmethod
    def calculate_optimal_parameters(v3pds_list, vws, num_it=100, lr=0.001, beta_1=0.9, beta_2=0.999):
        try:
            S=len(v3pds_list)
            K,q,Q_s,lambdas=CameraCalibration.initialization_of_parameters(v3pds_list=v3pds_list, vws=vws)

            optimizer=Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, eps=1e-4)
            history_L=np.zeros((num_it))
            history_norm_grad=np.zeros((num_it))
            x=CameraCalibration.get_x(K=K, q=q, Q_s=Q_s, lambdas=lambdas)
            for it in range(num_it):
                # print("It {} from {}".format(it + 1,num_it))
                L,grad=CameraCalibration.loss_function(x=x, v3pds_list=v3pds_list, vws=vws, S=S)
                x=optimizer.operate(params=x, grad=grad)
                history_L[it]=L
                history_norm_grad[it]=np.linalg.norm(grad)
            
            K_opt,q_opt,Q_s_opt,lambdas_opt,_=CameraCalibration.get_parameters(x=x, S=S)
            Qs_opt=CameraCalibration.get_Qs_from_Q_s(Q_s=Q_s_opt)
            # CameraCalibration.print_info_Twc(Qs=Qs_opt)

            return K_opt,q_opt,Qs_opt,lambdas_opt,history_L,history_norm_grad,True
        except np.linalg.LinAlgError as error:
            print("np.linalg.LinAlgError: {}".format(error))

            return None,None,None,None,None,None,False
        except Exception as error:
            print("Exception: {}".format(error))

            return None,None,None,None,None,None,False
