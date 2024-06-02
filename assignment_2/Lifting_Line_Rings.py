# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

def WakeGeometry(case, U_infty, omega, n_t, n_r, a_w, NBlades, R, chord, Twist, double=False, S_sep=0, phase_dif=0, plot=False):
        
    #Calculate intermediate parameters (Rings[controlpoint, Wake_point, Blade_Number, coordinate])
    if double:
        n_rotors = 2
        offset_y = S_sep/2
    else:
        n_rotors = 1
        offset_y = 0
    L = 2*n_t+2
    Rings = np.zeros([(len(R)-1), L, n_rotors*NBlades, 3])
    if case == 'turbine':
        U_w = U_infty*(1-a_w)
    else:
        U_w = U_infty*(1+a_w)

    
    #Blade geometry distribution
    r_U = R[1:].reshape([len(R)-1, 1, 1])
    r_B = R[:-1].reshape([len(R)-1, 1, 1])
    chord_U = chord[1:].reshape([len(R)-1, 1, 1])
    chord_B = chord[:-1].reshape([len(R)-1, 1, 1])
    twist_U = Twist[1:].reshape([len(R)-1, 1, 1]) * np.pi/180
    twist_B = Twist[:-1].reshape([len(R)-1, 1, 1]) * np.pi/180
    
    #Blade number
    Blades = np.arange(NBlades)
    Blades = Blades.reshape([1, 1, NBlades])
    theta_0 = (Blades) * 2*np.pi/NBlades
    if double:
        theta_0_2 = theta_0 - phase_dif*np.pi/180
        theta_0 = np.concatenate((theta_0, theta_0_2), axis=2)
        
        #Offset in y due to rotors being separated
        offset_y = offset_y * np.concatenate((np.ones([1, 1, NBlades]),-1*np.ones([1, 1, NBlades])),axis=2)
    else:
        offset_y = offset_y * np.ones([1, 1, NBlades])
        
    #Setup time array with extra zeros in middle
    t_end = n_r*2*np.pi/omega
    t_int = np.linspace(0,t_end,n_t)
    t_int = list(t_int)
    t_int.insert(0, 0)
    t = np.array(t_int)
    t = t.reshape([1, n_t+1, 1])
    
    #Calculate 4-dimensional matrix
    zero_offset = np.zeros([len(chord_U), 1, 1])
    
    offset_U = np.ones([1, n_t, 1])*chord_U.reshape([len(chord_U), 1, 1])
    offset_U = np.concatenate((zero_offset,offset_U),axis=1)
    
    offset_B = np.ones([1, n_t, 1])*np.fliplr(chord_B).reshape([len(chord_B), 1, 1])
    offset_B = np.concatenate((offset_B, zero_offset),axis=1)
    
    xw_U = t*U_w + offset_U*np.sin(twist_U)
    yw_U = -r_U*np.sin(omega*t+theta_0) - offset_U*np.cos(twist_U)*np.cos(theta_0) - offset_y
    zw_U = r_U*np.cos(omega*t+theta_0) - offset_U*np.cos(twist_U)*np.sin(theta_0)
    
    xw_B = np.fliplr(t)*U_w + offset_B*np.sin(twist_B)
    yw_B = -r_B*np.sin(omega*np.fliplr(t)+theta_0) - offset_B*np.cos(twist_B)*np.cos(theta_0) -  offset_y
    zw_B = r_B*np.cos(omega*np.fliplr(t)+theta_0) - offset_B*np.cos(twist_B)*np.sin(theta_0)
    
    xw = np.concatenate((xw_B, xw_U), axis=1)
    yw = np.concatenate((yw_B, yw_U), axis=1)
    zw = np.concatenate((zw_B, zw_U), axis=1)
            
    Rings[:,:,:,0] = xw
    Rings[:,:,:,1] = yw
    Rings[:,:,:,2] = zw
    
    #Plot limits
    xmin = -1
    xmax = t_end*U_w
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for Blade in range(NBlades*n_rotors):
            #x, y and z location of quarter chord
            Bladex_c_4 = np.zeros((len(R),))
            Bladey_c_4 = -R*np.sin(theta_0[0,0,Blade]) - offset_y[0,0,Blade]
            Bladez_c_4 = R*np.cos(theta_0[0,0,Blade])
            
            #x, y and z location of the left part of the blade
            Bladex_L = Bladex_c_4 - 0.25*chord*np.sin(Twist*np.pi/180.)
            Bladey_L = Bladey_c_4 + 0.25*chord*np.cos(Twist*np.pi/180.)*np.cos(theta_0[0,0,Blade])
            Bladez_L = Bladez_c_4 + 0.25*chord*np.cos(Twist*np.pi/180.)*np.sin(theta_0[0,0,Blade])
            
            #x, y and z location of the right part of the blade
            Bladex_R = Bladex_c_4 + 0.75*chord*np.sin(Twist*np.pi/180.)
            Bladey_R = Bladey_c_4 - 0.75*chord*np.cos(Twist*np.pi/180.)*np.cos(theta_0[0,0,Blade])
            Bladez_R = Bladez_c_4 - 0.75*chord*np.cos(Twist*np.pi/180.)*np.sin(theta_0[0,0,Blade])
            
            #Blade outline
            Bladex = np.hstack((Bladex_R, np.flip(Bladex_L), Bladex_R[0]))
            Bladey = np.hstack((Bladey_R, np.flip(Bladey_L), Bladey_R[0]))
            Bladez = np.hstack((Bladez_R, np.flip(Bladez_L), Bladez_R[0]))
            
            
            for Radial_Point in range(0,len(r_U),int(len(r_U)/2)-1):
                ax.plot(Rings[Radial_Point,:,Blade,0], Rings[Radial_Point,:,Blade,1], Rings[Radial_Point,:,Blade,2], 'b--', alpha = 0.75)
            
            ax.plot(Bladex_c_4,Bladey_c_4,Bladez_c_4, 'k--')
            ax.plot(Bladex, Bladey, Bladez, 'k-',linewidth = 1)
        ax.set_xlim([xmin, xmax])   
        if double:
            zmax = (S_sep + 2*R[-1])/2
            ax.set_zlim([-zmax,zmax])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    return Rings

# # =============================================================================
# #Defined elsewhere


# omega = 10
# U_infty = 100

# #Define input parameters
# n_t = 400
# n_r = 1
# a_w = 0
# Radius = 10

# NBlades = 6
# filename = "assignment_2/GeoOptimalpropeller.dat"
# Mat = np.loadtxt(filename, dtype='float',skiprows = 1)

# R = Mat[:,0]
# R_int = list(R)
# R_int.insert(0, R[0]-(R[5]-R[4])/2)
# R_int.insert(len(R_int),R[-1]+(R[5]-R[4])/2)
# R = np.array(R_int)

# Twist = Mat[:,1]
# Twist_int = list(Twist)
# Twist_int.insert(0, Twist[0]-(Twist[1]-Twist[0])/2)
# Twist_int.insert(len(Twist_int),Twist[-1]+(Twist[-1]-Twist[-2])/2)
# Twist = np.array(Twist_int)

# Chord = Mat[:,2]
# Chord_int = list(Chord)
# Chord_int.insert(0, Chord[0]-(Chord[1]-Chord[0])/2)
# Chord_int.insert(len(Chord_int),Chord[-1]+(Chord[-1]-Chord[-2])/2)
# Chord = np.array(Chord_int)

# WakeGeometry('propeller',U_infty, omega, n_t, n_r, a_w, NBlades, R*Radius, Chord, Twist, plot=True)
# plt.show()
# # =============================================================================
