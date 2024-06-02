import numpy as np

def induced_velocity_vortex_system(delta, BigMatrix, controlpoints, gamma, double=False, phase_dif=0, d_sep=0):
    NBlades = len(BigMatrix[0,0,:,0])
    Nwp = len(BigMatrix[0,:,0,0])
    Ncp_Blade = len(BigMatrix[:,0,0,0])
    # print(Ncp_Blade)
    Ncp = Ncp_Blade * NBlades
    Umatrix = np.zeros((Ncp,Ncp))
    Vmatrix = np.zeros((Ncp,Ncp))
    Wmatrix = np.zeros((Ncp,Ncp))
    
    #In the case of double rotor configuration
    if double:
        n_rotors = 2
        ref_matrix_phase = np.hstack((np.zeros((int(NBlades/2),)),np.ones((int(NBlades/2),))))
        ref_matrix_sep = np.hstack((np.ones((Ncp_Blade*int(NBlades/2),)),-1*np.ones((Ncp_Blade*int(NBlades/2),))))
    else:
        n_rotors = 1
        ref_matrix_phase = np.zeros((NBlades,))
        ref_matrix_sep = np.zeros((Ncp))
    
    #Define Gamma in the case of Gamma = 1
    gamma = np.ones([1,Ncp,1])*gamma

    #Create X1 and X2 $X1[controlpoint,jring,wakepoint]$
    X1_x = BigMatrix[:, :-1, 0, 0].reshape([1,Ncp_Blade,Nwp-1])
    X1_y = BigMatrix[:, :-1, 0, 1].reshape([1,Ncp_Blade,Nwp-1])
    X1_z = BigMatrix[:, :-1, 0, 2].reshape([1,Ncp_Blade,Nwp-1])
    
    X2_x = BigMatrix[:, 1:, 0, 0].reshape([1,Ncp_Blade,Nwp-1])
    X2_y = BigMatrix[:, 1:, 0, 1].reshape([1,Ncp_Blade,Nwp-1])
    X2_z = BigMatrix[:, 1:, 0, 2].reshape([1,Ncp_Blade,Nwp-1])
    
    #Create theta0 (for blade distribution over 2pi radians)
    phase = ref_matrix_phase*phase_dif*np.pi/180
    theta0 = np.ones([Ncp_Blade,1,1])*(0) * 2*np.pi/(NBlades/n_rotors)
    
    for i in range(1,NBlades):
        X1_x = np.concatenate((X1_x, BigMatrix[:, :-1, i, 0].reshape([1,Ncp_Blade,Nwp-1])),axis=1)
        X1_y = np.concatenate((X1_y, BigMatrix[:, :-1, i, 1].reshape([1,Ncp_Blade,Nwp-1])),axis=1)
        X1_z = np.concatenate((X1_z, BigMatrix[:, :-1, i, 2].reshape([1,Ncp_Blade,Nwp-1])),axis=1)
        
        X2_x = np.concatenate((X2_x, BigMatrix[:, 1:, i, 0].reshape([1,Ncp_Blade,Nwp-1])),axis=1)    
        X2_y = np.concatenate((X2_y, BigMatrix[:, 1:, i, 1].reshape([1,Ncp_Blade,Nwp-1])),axis=1)    
        X2_z = np.concatenate((X2_z, BigMatrix[:, 1:, i, 2].reshape([1,Ncp_Blade,Nwp-1])),axis=1)    
        
        theta0 = np.concatenate((theta0, np.ones([Ncp_Blade,1,1])*(i)*2*np.pi/(NBlades/n_rotors)-phase[i]),axis=0)
        
    #Create Xp $Xp[controlpoint,jring,wakepoint]$
    y_offset = (ref_matrix_sep*d_sep/2).reshape([Ncp,1,1])
    Xp_x = np.zeros([Ncp,1,1])
    Xp_y = -controlpoints.reshape([Ncp,1,1])*np.sin(theta0) - y_offset
    Xp_z = controlpoints.reshape([Ncp,1,1])*np.cos(theta0)
    
    [Umatrix, Vmatrix, Wmatrix] = velocity_matrix_from_vortex_filament(X1_x, X1_y, X1_z, X2_x, X2_y, X2_z, Xp_x, Xp_y, Xp_z, gamma, delta)
    return [np.matrix(Umatrix), np.matrix(Vmatrix), np.matrix(Wmatrix)]




def velocity_matrix_from_vortex_filament(X1_x, X1_y, X1_z, X2_x, X2_y, X2_z, Xp_x, Xp_y, Xp_z, gamma, delta):
    R1 = (np.multiply((Xp_x-X1_x),(Xp_x-X1_x))+np.multiply((Xp_y-X1_y),(Xp_y-X1_y))+np.multiply((Xp_z-X1_z),(Xp_z-X1_z)))**0.5 #5seconds
    R2 = (np.multiply((Xp_x-X2_x),(Xp_x-X2_x))+np.multiply((Xp_y-X2_y),(Xp_y-X2_y))+np.multiply((Xp_z-X2_z),(Xp_z-X2_z)))**0.5
    R_12_X = np.multiply((Xp_y-X1_y),(Xp_z-X2_z)) - np.multiply((Xp_z-X1_z),(Xp_y-X2_y))
    R_12_Y = -np.multiply((Xp_x-X1_x),(Xp_z-X2_z)) + np.multiply((Xp_z-X1_z),(Xp_x-X2_x))
    R_12_Z = np.multiply((Xp_x-X1_x),(Xp_y-X2_y)) - np.multiply((Xp_y-X1_y),(Xp_x-X2_x))
    R_12_sqr = np.multiply(R_12_X,R_12_X) + np.multiply(R_12_Y,R_12_Y) + np.multiply(R_12_Z,R_12_Z)
    R0_1 = (X2_x - X1_x)*(Xp_x - X1_x) + (X2_y - X1_y)*(Xp_y - X1_y) + (X2_z - X1_z)*(Xp_z - X1_z)
    R0_2 = (X2_x - X1_x)*(Xp_x - X2_x) + (X2_y - X1_y)*(Xp_y - X2_y) + (X2_z - X1_z)*(Xp_z - X2_z)
    
    dl = np.ones([len(R_12_sqr[:,0,0]),1,1])*np.sqrt((X2_x-X1_x)**2 + (X2_y-X1_y)**2 + (X2_z-X1_z)**2 )
    
    #If controlpoint lies inside vortex filament, use solid body rotation
#    n1 = np.argwhere(R_12_sqr < (delta*dl**2)**2)
#    n2 = np.argwhere(R1 < delta)
#    n3 = np.argwhere(R2 < delta)
#    R_12_sqr[n1[:,0],n1[:,1],n1[:,2]] = (delta*dl[n1[:,0],n1[:,1],n1[:,2]]**2)**2
#    R1[n2[:,0],n2[:,1],n2[:,2]] = delta*dl[n2[:,0],n2[:,1],n2[:,2]]
#    R2[n3[:,0],n3[:,1],n3[:,2]] = delta*dl[n3[:,0],n3[:,1],n3[:,2]]
    n1 = np.argwhere(R_12_sqr < (delta*dl)**2)
    n2 = np.argwhere(R1 < delta)
    n3 = np.argwhere(R2 < delta)
    R_12_sqr[n1[:,0],n1[:,1],n1[:,2]] = (delta*dl[n1[:,0],n1[:,1],n1[:,2]])**2
    R1[n2[:,0],n2[:,1],n2[:,2]] = delta*dl[n2[:,0],n2[:,1],n2[:,2]]
    R2[n3[:,0],n3[:,1],n3[:,2]] = delta*dl[n3[:,0],n3[:,1],n3[:,2]]

    #Calculate induced velocities
    K = gamma / (4*np.pi*R_12_sqr)*(R0_1/R1 - R0_2/R2)
    U = np.multiply(K,R_12_X)
    V = np.multiply(K,R_12_Y)
    W = np.multiply(K,R_12_Z)
    
    return [np.sum(U,axis=2), np.sum(V,axis=2), np.sum(W,axis=2)]


# import matplotlib.pyplot as plt
# n=201

# X1_x = np.ones([1,1,1])*-5
# X1_y = np.zeros([1,1,1])
# X1_z = np.zeros([1,1,1])

# X2_y = np.ones([1,1,1]) *4
# X2_x = np.ones([1,1,1])*5
# X2_z = np.zeros([1,1,1])

# delta = 0.05
# dl = float(X2_x-X1_x)

# Xp_x = np.zeros([n,1,1])
# Xp_y = np.linspace(0,1,n).reshape([n,1,1])*dl
# Xp_z = np.zeros([n,1,1])



# [u,v,w] = velocity_matrix_from_vortex_filament(X1_x, X1_y, X1_z, X2_x, X2_y, X2_z, Xp_x, Xp_y, Xp_z, 1, delta)
# R = Xp_y.reshape((n,))/dl

# plt.plot(R,u)
# plt.plot(R,v)
# plt.plot(R,w)
# plt.show()