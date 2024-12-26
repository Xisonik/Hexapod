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
    AB_i_pow_2 = 0
    print("i = ", i)

    print("a,b:", B[6*0+i],A[6*0+i])
    AB_i_pow_2 += np.power(B[6*0+i] - A[6*0+i], 2)
    print("a,b:", B[6*1+i],A[6*1+i])
    AB_i_pow_2 += np.power(B[6*1+i] - A[6*1+i], 2)
    print("a,b:", B[6*2+i],A[6*2+i])
    AB_i_pow_2 += np.power(B[6*2+i] - A[6*2+i], 2)
    print("AB_i^2: ", AB_i_pow_2)

    return np.sqrt(AB_i_pow_2)

def find_alpha_i(B, B_last, i):
    print("check_2", np.arctan((B[2*6 + i] - B_last[2*6 + i])/B[3*6 + i]))
    return np.arctan((B[2*6 + i] - B_last[2*6 + i])/B[3*6 + i])

# def find_T(x, y, z, q1=0, q2=0, q3=0):
#     T = np.empty(shape=(16))

#     T[0] = np.cos(q3)*np.cos(q2)
#     T[1] = np.cos(q3)*np.sin(q2)*np.sin(q1) - np.sin(q3)*np.cos(q1)
#     T[2] = np.cos(q3)*np.sin(q2)*np.cos(q1) + np.sin(q3)*np.sin(q1)

#     T[3] = np.sin(q3)*np.cos(q2)
#     T[4] = np.sin(q3)*np.sin(q2)*np.sin(q1) + np.cos(q3)*np.cos(q1)
#     T[5] = np.sin(q3)*np.sin(q2)*np.cos(q1) - np.cos(q3)*np.sin(q1)

#     T[6] = -np.sin(q2)
#     T[7] = np.cos(q2) * np.sin(q1)
#     T[8] = np.cos(q2) * np.cos(q1)

#     T[9] = x
#     T[10] = y
#     T[11] = z

#     T[12] = 0
#     T[13] = 0
#     T[14] = 0
#     T[15] = 1
#     print("T ", T)

#     return T

def find_T(x, y, z, q1=0, q2=0, q3=0):
    T = np.empty(shape=(16))

    T[0] = np.cos(q3)*np.cos(q2)
    T[1] = np.cos(q3)*np.sin(q2)*np.sin(q1) - np.sin(q3)*np.cos(q1)
    T[2] = np.cos(q3)*np.sin(q2)*np.cos(q1) + np.sin(q3)*np.sin(q1)
    T[3] = x

    T[4] = np.sin(q3)*np.cos(q2)
    T[5] = np.sin(q3)*np.sin(q2)*np.sin(q1) + np.cos(q3)*np.cos(q1)
    T[6] = np.sin(q3)*np.sin(q2)*np.cos(q1) - np.cos(q3)*np.sin(q1)
    T[7] = y

    T[8] = -np.sin(q2)
    T[9] = np.cos(q2) * np.sin(q1)
    T[10] = np.cos(q2) * np.cos(q1)
    T[11] = z

    T[12] = 0
    T[13] = 0
    T[14] = 0
    T[15] = 1
    print("T ", T)

    return T

# def find_T(x, y, z, q1=0, q2=0, q3=0):
#     T = np.empty(shape=(16))

#     T[0] = np.cos(q3)*np.cos(q2)
#     T[1] = np.sin(q3)*np.cos(q2)
#     T[2] = -np.sin(q2)
#     T[3] = 0

#     T[4] = np.cos(q3)*np.sin(q2)*np.sin(q1) - np.sin(q3)*np.cos(q1)
#     T[5] = np.sin(q3)*np.sin(q2)*np.sin(q1) + np.cos(q3)*np.cos(q1)
#     T[6] = np.cos(q2) * np.sin(q1)
#     T[7] = 0
    
#     T[8] = np.cos(q3)*np.sin(q2)*np.cos(q1) + np.sin(q3)*np.sin(q1)
#     T[9] = np.sin(q3)*np.sin(q2)*np.cos(q1) - np.cos(q3)*np.sin(q1)
#     T[10] = np.cos(q2) * np.cos(q1)
#     T[11] = 0

#     T[12] = x
#     T[13] = y
#     T[14] = z    
#     T[15] = 1
#     print("T ", T)

#     return T
# main function
if __name__ == "__main__":
    H = Hexapod()
    A = np.empty(shape=(24))

    A[0] = H.R*np.cos(H.phi_b)
    A[1] = H.R*np.cos(2*np.pi/3 - H.phi_b)
    A[2] = H.R*np.cos(2*np.pi/3 + H.phi_b)  
    A[3] = H.R*np.cos(4*np.pi/3 - H.phi_b)
    A[4] = H.R*np.cos(4*np.pi/3 + H.phi_b)
    A[5] = H.R*np.cos(-H.phi_b)
    A[6] = H.R*np.sin(H.phi_b)
    A[7] = H.R*np.sin(2*np.pi/3 - H.phi_b)
    A[8] = H.R*np.sin(2*np.pi/3 + H.phi_b)
    A[9] = H.R*np.sin(4*np.pi/3 - H.phi_b)
    A[10] = H.R*np.sin(4*np.pi/3 + H.phi_b)
    A[11] = H.R*np.sin(-H.phi_b)
    A[12] = H.h_b
    A[13] = H.h_b
    A[14] = H.h_b
    A[15] = H.h_b
    A[16] = H.h_b
    A[17] = H.h_b
    A[18] = 1
    A[19] = 1
    A[20] = 1
    A[21] = 1
    A[22] = 1
    A[23] = 1

    B = np.empty(shape=(24))
    B[0] = H.r*np.cos(H.phi_mp)
    B[1] = H.r*np.cos(2*np.pi/3 - H.phi_mp)
    B[2] = H.r*np.cos(2*np.pi/3 + H.phi_mp)
    B[3] = H.r*np.cos(4*np.pi/3 - H.phi_mp)
    B[4] = H.r*np.cos(4*np.pi/3 + H.phi_mp)
    B[5] = H.r*np.cos(-H.phi_mp)
    B[6] = H.r*np.sin(H.phi_mp)
    B[7] = H.r*np.sin(2*np.pi/3 - H.phi_mp)
    B[8] = H.r*np.sin(2*np.pi/3 + H.phi_mp)
    B[9] = H.r*np.sin(4*np.pi/3 - H.phi_mp)
    B[10] = H.r*np.sin(4*np.pi/3 + H.phi_mp)
    B[11] = H.r*np.sin(-H.phi_mp)
    B[12] = -H.h_mp
    B[13] = -H.h_mp
    B[14] = -H.h_mp
    B[15] = -H.h_mp
    B[16] = -H.h_mp
    B[17] = -H.h_mp
    B[18] = 1
    B[19] = 1
    B[20] = 1
    B[21] = 1
    B[22] = 1
    B[23] = 1

    T = np.zeros(shape=(16))
    theta = np.zeros(6)
    theta_for_servo = np.zeros(6)
    servo = np.zeros(6)

    z = 254
    x = y = 0
    l_0 = 220
    R = 17.5
    MAX_ANGLE_X = np.radians(5)
    MAX_ANGLE_Y = np.radians(5)
    B_last = np.zeros(24)
    B_last[0]=B[0]
    B_last[1]=B[1]
    B_last[2]=B[2]
    B_last[3]=B[3]
    B_last[4]=B[4]
    B_last[5]=B[5]
    B_last[6]=B[6]
    B_last[7]=B[7]
    B_last[8]=B[8]
    B_last[9]=B[9]
    B_last[10]=B[10]
    B_last[11]=B[11]
    B_last[12]=B[12]
    B_last[13]=B[13]
    B_last[14]=B[14]
    B_last[15]=B[15]
    B_last[16]=B[16]
    B_last[17]=B[17]
    B_last[18]=B[18]
    B_last[19]=B[19]
    B_last[20]=B[20]
    B_last[21]=B[21]
    B_last[22]=B[22]
    B_last[23]=B[23]

    print("matrix", A, B)
    T = find_T(x, y, z, MAX_ANGLE_X, MAX_ANGLE_Y)
    B[0] = T[0]*B_last[0] + T[1]*B_last[6] + T[2]*B_last[12] + T[3]*B_last[18]
    B[1] = T[0]*B_last[1] + T[1]*B_last[7] + T[2]*B_last[13] + T[3]*B_last[19]
    B[2] = T[0]*B_last[2] + T[1]*B_last[8] + T[2]*B_last[14] + T[3]*B_last[20]
    B[3] = T[0]*B_last[3] + T[1]*B_last[9] + T[2]*B_last[15] + T[3]*B_last[21]
    B[4] = T[0]*B_last[4] + T[1]*B_last[10] + T[2]*B_last[16] + T[3]*B_last[22]
    B[5] = T[0]*B_last[5] + T[1]*B_last[11] + T[2]*B_last[17] + T[3]*B_last[23]
    B[6] = T[4]*B_last[0] + T[5]*B_last[6] + T[6]*B_last[12] + T[7]*B_last[18]
    B[7] = T[4]*B_last[1] + T[5]*B_last[7] + T[6]*B_last[13] + T[7]*B_last[19]
    B[8] = T[4]*B_last[2] + T[5]*B_last[8] + T[6]*B_last[14] + T[7]*B_last[20]
    B[9] = T[4]*B_last[3] + T[5]*B_last[9] + T[6]*B_last[15] + T[7]*B_last[21]
    B[10] = T[4]*B_last[4] + T[5]*B_last[10] + T[6]*B_last[16] + T[7]*B_last[22]
    B[11] = T[4]*B_last[5] + T[5]*B_last[11] + T[6]*B_last[17] + T[7]*B_last[23]
    B[12] = T[8]*B_last[0] + T[9]*B_last[6] + T[10]*B_last[12] + T[11]*B_last[18]
    B[13] = T[8]*B_last[1] + T[9]*B_last[7] + T[10]*B_last[13] + T[11]*B_last[19]
    B[14] = T[8]*B_last[2] + T[9]*B_last[8] + T[10]*B_last[14] + T[11]*B_last[20]
    B[15] = T[8]*B_last[3] + T[9]*B_last[9] + T[10]*B_last[15] + T[11]*B_last[21]
    B[16] = T[8]*B_last[4] + T[9]*B_last[10] + T[10]*B_last[16] + T[11]*B_last[22]
    B[17] = T[8]*B_last[5] + T[9]*B_last[11] + T[10]*B_last[17] + T[11]*B_last[23]
    B[18] = T[12]*B_last[0] + T[13]*B_last[6] + T[14]*B_last[12] + T[15]*B_last[18]
    B[19] = T[12]*B_last[1] + T[13]*B_last[7] + T[14]*B_last[13] + T[15]*B_last[19]
    B[20] = T[12]*B_last[2] + T[13]*B_last[8] + T[14]*B_last[14] + T[15]*B_last[20]
    B[21] = T[12]*B_last[3] + T[13]*B_last[9] + T[14]*B_last[15] + T[15]*B_last[21]
    B[22] = T[12]*B_last[4] + T[13]*B_last[10] + T[14]*B_last[16] + T[15]*B_last[22]
    B[23] = T[12]*B_last[5] + T[13]*B_last[11] + T[14]*B_last[17] + T[15]*B_last[23]
    print("B_new: " , B)
    flag_valid = 0
    # for k in range(0,6):#убрать цикл
    phi_start = 85
    theta[0] = (find_theta_i_new(A, B, H, 0) - l_0)/R
    theta_for_servo[0] = phi_start - np.degrees(theta[0])
    if not((theta[0] > -np.pi/2) and (theta[0] < np.pi/2)) or np.isnan(theta[0]):
        flag_valid = 1
    theta[1] = (find_theta_i_new(A, B, H, 1) - l_0)/R
    theta_for_servo[1] = phi_start - np.degrees(theta[1])
    if not((theta[1] > -np.pi/2) and (theta[1] < np.pi/2)) or np.isnan(theta[1]):
        flag_valid = 1
    theta[2] = (find_theta_i_new(A, B, H, 2) - l_0)/R
    theta_for_servo[2] = phi_start - np.degrees(theta[2])
    if not((theta[2] > -np.pi/2) and (theta[2] < np.pi/2)) or np.isnan(theta[2]):
        flag_valid = 1
    theta[3] = (find_theta_i_new(A, B, H, 3) - l_0)/R
    theta_for_servo[3] = phi_start - np.degrees(theta[3])
    if not((theta[3] > -np.pi/2) and (theta[3] < np.pi/2)) or np.isnan(theta[3]):
        flag_valid = 1
    theta[4] = (find_theta_i_new(A, B, H, 4) - l_0)/R
    theta_for_servo[4] = phi_start - np.degrees(theta[4])
    if not((theta[4] > -np.pi/2) and (theta[4] < np.pi/2)) or np.isnan(theta[4]):
        flag_valid = 1
    theta[4] = (find_theta_i_new(A, B, H, 4) - l_0)/R
    theta_for_servo[4] = phi_start - np.degrees(theta[4])
    if not((theta[4] > -np.pi/2) and (theta[4] < np.pi/2)) or np.isnan(theta[4]):
        flag_valid = 1

        
    servo[0] = theta_for_servo[0]
    servo[1] = theta_for_servo[5]
    servo[2] = theta_for_servo[4]
    servo[3] = theta_for_servo[3]
    servo[4] = theta_for_servo[2]
    servo[5] = theta_for_servo[1]
    if flag_valid:
        print("ERROR")
    print("l0 = ", l_0)
    print("z = ", z)
    print("alpha, betta = ", MAX_ANGLE_X, MAX_ANGLE_Y)
    print("servo", servo)
