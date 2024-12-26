import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator

class Hexapod:
    def __init__(self):
        self.R = 10.3 * 10
        self.r = 7.5* 10
        self.h_b = 2.5* 10
        self.h_mp = 2.1* 10
        self.phi_b = np.radians(30)
        self.phi_mp = np.radians(5)
        
        

def find_theta_i_new(A, B, H, i):
    print("i = ", i)
    AB_i_pow_2 = 0
    for k in range(3):
        print("a,b:", B[k][i], A[k][i])
        AB_i_pow_2 += np.power(B[k][i] - A[k][i], 2)
    print("AB_i^2: ", AB_i_pow_2)
    return np.sqrt(AB_i_pow_2)

def find_alpha_i(B, B_last, i):
    print("check_2", np.arctan((B[2][i] - B_last[2][i])/B[3][i]))
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
    print("T ", T)

    return T

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
    print("matrix", A, B)
    T = np.zeros(shape=(4,4))
    theta = np.zeros(6)
    theta_for_servo = np.zeros(6)
    servo = np.zeros(6)

    z = 254#235.1915
    x = y = 0
    MAX_ANGLE_X = np.radians(5)
    MAX_ANGLE_Y = np.radians(5)
    T = find_T(x, y, z, MAX_ANGLE_X, MAX_ANGLE_Y)
    B_last=B
    B = np.matmul(T, B_last)

    flag_valid = 0
    for k in range(0,6):
        l_0 = 220#195
        R = 17.5
        # theta[k] = find_theta_i_new(A, B, H, k) - find_alpha_i(B, B_last, k)
        phi_start = 85
        print(find_theta_i_new(A, B, H, k))
        theta[k] = (find_theta_i_new(A, B, H, k) - l_0)/R
        print("delta ", k, np.degrees(theta[k]))
        theta_for_servo[k] = phi_start - np.degrees(theta[k])
        
        if not((theta[k] > -np.pi/2) and (theta[k] < np.pi/2)) or np.isnan(theta[k]):
            flag_valid = 1

        
    servo[0] = theta_for_servo[0]
    servo[1] = theta_for_servo[5]
    servo[2] = theta_for_servo[4]
    servo[3] = theta_for_servo[3]
    servo[4] = theta_for_servo[2]
    servo[5] = theta_for_servo[1]
    print(A)
    print(B_last)
    print(B)
    if flag_valid:
        print("ERROR")
    print("l0 = ", l_0)
    print("z = ", z)
    print("alpha, betta = ", MAX_ANGLE_X, MAX_ANGLE_Y)
    print("servo", servo)
