
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def newGamma(case, rho, U_infty, u, v, w, Omega, controlpoints, BigMatrix, twist, polar_alpha, polar_cl, chord, double=False, phase_dif=0):
    """
    calculates the new circulation distribution along the blade
    """
    if case == 'turbine':
        Vaxial = U_infty - u
    else:
        Vaxial = U_infty - u
    if double:
        n_rotors = 2
    else:
        n_rotors = 1
    Blades = np.arange(len(BigMatrix[0,0,:,0]))
    theta_0 = (Blades) * 2*np.pi/(len(Blades)/n_rotors)
    if double:
        theta_0_2 = theta_0 - phase_dif*np.pi/180
        theta_0 = np.hstack((theta_0, theta_0_2))
    
    Vtan =  np.mat(np.zeros([len(controlpoints)*len(Blades), 1]))
    for i in range(len(controlpoints)):
        for j in range(len(Blades)):
            i_cp = j*len(controlpoints)+i
            n_times_vt = +np.cos(theta_0[j])*v[i_cp] + np.sin(theta_0[j])*w[i_cp]
            if case == 'turbine':
                Vtan[i_cp] = Omega*controlpoints[i] + n_times_vt
            else:
                Vtan[i_cp] = Omega*controlpoints[i] + n_times_vt
    Vp = np.sqrt(np.multiply(Vaxial, Vaxial) + np.multiply(Vtan, Vtan))
    inflowangle = np.arctan2(Vaxial, Vtan)
    alpha = inflowangle*180/np.pi + twist
    s_cl = InterpolatedUnivariateSpline(polar_alpha, polar_cl, k=1)
    cl = s_cl(alpha)
    gamma = 0.5*np.multiply(np.multiply(Vp,cl),chord)
        
    return gamma