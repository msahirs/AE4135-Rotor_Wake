
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

def loadBladeOverAllElements(case, rho, U_infty, u, v, w, Omega, controlpoints, twist, polar_alpha, polar_cl, polar_cd, chord, BigMatrix, double=False, phase_dif=0):
    """
    calculates the loads on all blade elements in a single blade
    """
    #First determine velocities
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
    
    #Next determine force coefficients on the blade elements
    inflowangle = np.arctan2(Vaxial,Vtan)
    alpha = inflowangle*180/np.pi+twist
    s_cl = InterpolatedUnivariateSpline(polar_alpha, polar_cl, k=1)
    s_cd = InterpolatedUnivariateSpline(polar_alpha, polar_cd, k=1)
    cl = s_cl(alpha)
    cd = s_cd(alpha)
    
    #From the force coefficients determine the lift and drag
    lift = 0.5*rho*np.multiply(np.multiply(np.multiply(Vp,Vp),cl),chord)
    drag = 0.5*rho*np.multiply(np.multiply(np.multiply(Vp,Vp),cd),chord)
    fnorm = np.multiply(lift,np.cos(inflowangle))+np.multiply(drag,np.sin(inflowangle))
    ftan = np.multiply(lift,np.sin(inflowangle))-np.multiply(drag,np.cos(inflowangle))
    
    return [fnorm, ftan, alpha, inflowangle, Vaxial, Vtan]