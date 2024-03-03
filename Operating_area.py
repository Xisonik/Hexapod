import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class Hexapod:
    def __init__(self):
        """
        self.R = 16.15 * 10
        self.r = 8* 10
        self.h_b = 4.35* 10
        self.h_mp = 0.85* 10
        self.l1 = 3.75* 10
        self.l2 = 15* 10
        self.phi_b = np.pi/6#np.radians(0) #-20
        self.phi_mp = np.pi/18#np.radians(0) #45
        self.r_lower =np.radians(10)#10
        self.r_upper =np.radians(120)#120
        self.delta = 2*np.pi/3
        
        """
        self.R = 170#16.15 * 10
        self.r = 160#8* 10
        self.h_b = 45#4.35* 10
        self.h_mp = 15#0.85* 10
        self.l1 = 68#3.75* 10
        self.l2 = 168#15* 10
        self.phi_b = np.pi/6#np.radians(0) #-20
        self.phi_mp = np.pi/18#np.radians(0) #45
        self.r_lower =np.radians(10)#10
        self.r_upper =np.radians(120)#120
        self.delta = 2*np.pi/3
        

def find_theta_i_new(A, B, H, i):
    AB_i_pow_2 = 0
    for k in range(3):
        AB_i_pow_2 += np.power(B[k][i] - A[k][i], 2)
    
    #print("a", AB_i_pow_2)
    cos_q = (np.power(H.l1, 2) + AB_i_pow_2 - np.power(H.l2, 2))/(2*H.l1*np.sqrt(AB_i_pow_2))
    
    #print(cos_q)
    return np.arccos(np.power(H.l1, 2) + AB_i_pow_2 - np.power(H.l2, 2))/(2*H.l1*np.sqrt(AB_i_pow_2))

def find_alpha_i(B, B_last, i):
    return np.arctan((B[2][i] - B_last[2][i])/B[3][i])

def find_T(x, y, z, q1=0, q2=0, q3=0):
    T = np.empty(shape=(4,4))

    T[0][0] = np.cos(q3)*np.cos(q2)
    T[0][1] = np.cos(q3)*np.sin(q2)*np.sin(q1) - np.sin(q3)*np.cos(q1)
    T[0][2] = np.cos(q3)*np.sin(q2)*np.cos(q1) + np.sin(q3)*np.sin(q1)

    T[1][0] = np.sin(q3)*np.cos(q2)
    T[1][1] = np.sin(q3)*np.sin(q2)*np.sin(q1) + np.cos(q3)*np.cos(q1)
    T[1][2] = np.sin(q3)*np.sin(q2)*np.cos(q1) - np.cos(q3)*np.sin(q1)

    T[2][0] = -np.sin(q2)
    T[2][1] = np.cos(q2) * np.sin(q1)
    T[2][2] = np.cos(q2) * np.cos(q1)

    T[0][3] = x
    T[1][3] = y
    T[2][3] = z

    T[3][0] = 0
    T[3][1] = 0
    T[3][2] = 0
    T[3][3] = 1

    return T



def find_operating_area(A, B_last, H, MAX_ANGLE_X=0, MAX_ANGLE_Y=0, MAX_ANGLE_Z=0):
    T = np.zeros(shape=(4,4))
    h = 0.5
    z_start = 20
    theta = np.zeros(6)
    z = z_start
    area = np.zeros(2)
    flag_srart = 0
    while(z < H.l1+H.l2+H.h_b + H.h_mp):
        flag_valid = 1
        z += h
        x = y = 0
        T = find_T(x, y, z, MAX_ANGLE_X, MAX_ANGLE_Y, MAX_ANGLE_Z)
        B = np.matmul(T, B_last)

        for k in range(0,6):
            sign = np.sign(y)
            if sign == 0:
                sign = -1
            #sign * np.power(-1, k//3)*
            theta[k] = find_theta_i_new(A, B, H, k) - find_alpha_i(B, B_last, k)
            if not((theta[k] > -np.pi/2) and (theta[k] < np.pi/2)) or np.isnan(theta[k]):
                flag_valid = 0

        print(theta)
        if flag_valid:
            if flag_srart == 0:
                flag_srart += 1
                area[0] = z
            area[1] = z

        print("z = ", z)
        #print(np.rad2deg(theta))
        
    return area


# main function
if __name__ == "__main__":
    H = Hexapod()
    A = np.empty(shape=(4,6))
    A[0][0] = H.R*np.cos(H.phi_b)
    A[1][0] = H.R*np.sin(H.phi_b)
    
    A[0][1] = H.R*np.cos(2*np.pi/3 - H.phi_b)
    A[1][1] = H.R*np.sin(2*np.pi/3 - H.phi_b)

    A[0][2] = H.R*np.cos(2*np.pi/3 + H.phi_b)
    A[1][2] = H.R*np.sin(2*np.pi/3 + H.phi_b)

    A[0][3] = H.R*np.cos(4*np.pi/3 - H.phi_b)
    A[1][3] = H.R*np.sin(4*np.pi/3 - H.phi_b)

    A[0][4] = H.R*np.cos(4*np.pi/3 + H.phi_b)
    A[1][4] = H.R*np.sin(4*np.pi/3 + H.phi_b)

    A[0][5] = H.R*np.cos(-H.phi_b)
    A[1][5] = H.R*np.sin(-H.phi_b)
    
    for i in range(6):
        A[2][i] = H.h_b
        A[3][i] = 1

    B = np.empty(shape=(4,6))
    B[0][0] = H.r*np.cos(H.phi_mp)
    B[1][0] = H.r*np.sin(H.phi_mp)
    
    B[0][1] = H.r*np.cos(2*np.pi/3 - H.phi_mp)
    B[1][1] = H.r*np.sin(2*np.pi/3 - H.phi_mp)

    B[0][2] = H.r*np.cos(2*np.pi/3 + H.phi_mp)
    B[1][2] = H.r*np.sin(2*np.pi/3 + H.phi_mp)

    B[0][3] = H.r*np.cos(4*np.pi/3 - H.phi_mp)
    B[1][3] = H.r*np.sin(4*np.pi/3 - H.phi_mp)

    B[0][4] = H.r*np.cos(4*np.pi/3 + H.phi_mp)
    B[1][4] = H.r*np.sin(4*np.pi/3 + H.phi_mp)

    B[0][5] = H.r*np.cos(-H.phi_mp)
    B[1][5] = H.r*np.sin(-H.phi_mp)
    
    for i in range(6):
        B[2][i] = -H.h_mp
        B[3][i] = 1


    print(A)
    print(B)
    #MAX_ANGLE = np.radians(20)
    NUM_OF_MAX_ANGLE = 1
    z_min = np.zeros(shape=(NUM_OF_MAX_ANGLE, NUM_OF_MAX_ANGLE))
    z_max = np.zeros(shape=(NUM_OF_MAX_ANGLE, NUM_OF_MAX_ANGLE))
    len = np.zeros(NUM_OF_MAX_ANGLE)
    
    for MAX_ANGLE_X in range(0, NUM_OF_MAX_ANGLE):
        for MAX_ANGLE_Y in range(0, NUM_OF_MAX_ANGLE):
            z_min[MAX_ANGLE_X][MAX_ANGLE_Y], z_max[MAX_ANGLE_X][MAX_ANGLE_Y] = find_operating_area(A, B, H, np.radians(MAX_ANGLE_X), np.radians(MAX_ANGLE_Y))

    print(z_min)
    print(z_max)

    #plot_verticles(vertices = z_min, isosurf = False)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(0, NUM_OF_MAX_ANGLE, 1)
    Y = np.arange(0, NUM_OF_MAX_ANGLE, 1)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = z_min

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    surf = ax.plot_surface(X, Y, z_max, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 300)
    plt.xlabel("alpha")
    plt.ylabel("betta")
    plt.title("Рабочая зона гексапода z(alpha, betta)")
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

"""
"""