
from scipy.interpolate import InterpolatedUnivariateSpline
import time
import numpy as np

from GammaFunction import newGamma as GF
from Lifting_Line_Loads import loadBladeOverAllElements as LBE
from Lifting_Line_Rings import WakeGeometry as WG
from Vortex_Ring_System import induced_velocity_vortex_system as IV
import matplotlib.pyplot as plt

import pandas as pd

t_start = time.time()
#-------------------------------------------------------
#------(Start) Setup Block: Can be altered by user------
#-------------------------------------------------------

factor = 1
delta = 0.04
#Blade discretisation parameters
Ncp = 10
dis_mode = 'constant' # cosine
# reload = True
order = 3

#Vortex wake parameters
n_r = 30
n_pr = 40
n_r_list = [1,2,3,4,5,10,25,50,100]
lines = ['-x','-o','-s','-d','-^','--x','--o','--s','--d','--^']
a_w_init = 0.3
case = 'propeller'             #Choice between 'turbine' and 'propeller'

#Iteration parameters
n_iterations = 10
Error_margin = 0.001

#Plot settings
FontSize = 22

#Program modes
Sensitivity = False          #Only runs if Double_rotor and BEM_compare == False

Plot = True
show_results_BEM = False
plot_Wake = False

#Model setup for double rotor configuration
Double_rotor = False         #Set to True for double rotor configuration
L_sep = 1                    #Separation distance between two rotors expressed
                             #in rotor diameter
phase_dif = 0                #Phase difference in degrees

#-----------------------------------------------
#-------------(End) Setup Block-----------------
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#--User should NOT alter any code hereonafter---
#-----------------------------------------------
#------(Start) Initialise problem setup---------
#-----------------------------------------------


import Geometry_propeller as V
Omega = V.rpm/60*2*np.pi
airfoil = 'assignment_2/'+'ARA_airfoil_data.xlsx'
sheet   = "Sheet1"

#Pre-calculations in case of double rotor configuration
if Double_rotor:
    BEM_compare = False
    L_sep = L_sep*(V.R*2)
    n_rotors = 2
else:
    n_rotors = 1

#Load in blade geometry
Geo_filename = 'GeoOptimal'+case+'.dat'
Geo = np.loadtxt("assignment_2/"+Geo_filename , dtype='float',skiprows = 1)
N_ann = len(Geo[:,0])


# 

#Continue with setting up lifting line model
controlpoints = Geo[:,0].reshape((len(Geo[:,0]),1))[::2]
Twist_cp = Geo[:,1].reshape(len(Geo[:,1]),1)[::2]
Chord_cp = Geo[:,2].reshape(len(Geo[:,2]),1)[::2]



#Determine chord, twist and radii at discretised locations
#The matrix Geo only yields the geometry at the controlpoints, which lie inside
#the discretised locations
R0 = V.rootradius_R
Rend = V.tipradius_R
if dis_mode == 'constant':
    R_disL = np.linspace(R0, Rend, Ncp+1)
elif dis_mode == 'cosine':
    R_mid = (Rend-R0)/2 
    R_disL = R0 + R_mid*(1-np.cos(np.pi/Ncp*np.arange(Ncp+1))) 

s_chord = InterpolatedUnivariateSpline(controlpoints, Chord_cp, k=order)
Chord_disL = s_chord(R_disL)
s_twist = InterpolatedUnivariateSpline(controlpoints, Twist_cp, k=order)
Twist_disL = s_twist(R_disL)


#Create twist, chord and controlpoint vectors that contain the controlpoints on
#all the blades
controlpoints_all = controlpoints
Twist_all_cp = Twist_cp
chord_all_cp = Chord_cp
for i in range(1,V.Nblades*n_rotors):
    controlpoints_all = np.vstack((controlpoints_all, controlpoints))
    Twist_all_cp = np.vstack((Twist_all_cp,Twist_cp))
    chord_all_cp = np.vstack((chord_all_cp,Chord_cp))

#Load in polar data
data1=pd.read_excel(airfoil, sheet)
polar_alpha = data1['Alfa']
polar_cl = data1['Cl']
polar_cd = data1['Cd']

#Define total number of points in on lifting line wake
if not Sensitivity:
    n_t = n_r*n_pr
else:
    n_t_lst = np.array(n_r_list)*n_pr
    CT_sens = np.zeros((len(n_t_lst),1))
    CP_sens = np.zeros((len(n_t_lst),1))
    Gamma_sens = np.zeros((len(n_t_lst),Ncp))
    Fnorm_sens = np.zeros((len(n_t_lst),Ncp))
    Ftan_sens = np.zeros((len(n_t_lst),Ncp))
print('Lifting line method for '+case+' has started')
    
#Figure settings
plt.rcParams.update({'font.size': FontSize})
#-----------------------------------------------
#-------(End) Initialise problem setup----------
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#-(Start) Main block 1: Initialise circulation--
#-----------------------------------------------
if not Sensitivity:
    #Setup initial vortex wake structure
    t_VW_0 = time.time()
    if Double_rotor:
        Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w_init, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL, double=True, S_sep=L_sep, phase_dif=phase_dif, plot=plot_Wake)
    else:
        Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w_init, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL, plot=plot_Wake)
    
    
    t_VW_end = time.time()
    print('Vortex wake geometry is determined in ', t_VW_end-t_VW_0,' seconds')
    
    #Setup Biot-Savart induction matrix for Gamma = 1
    if case == 'propeller':
        Gamma_1 = -1
    else:
        Gamma_1 = 1
        
    if Double_rotor:
        [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, Gamma_1, double=True, phase_dif=phase_dif, d_sep=L_sep)
    else:
        [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, Gamma_1)
    t_ind_end = time.time()
    print('Induced velocity matrices are calculated in ',t_ind_end-t_VW_end,' seconds')
    
    #Initial estimate for circulation (with zero induced velocity)
    t_Gamma_0 = time.time()
    Uind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
    Vind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
    Wind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
    if Double_rotor:
        Gamma = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp, double=True, phase_dif=phase_dif)
    else:
        Gamma = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp)
    t_Gamma_end = time.time()
    print('Gamma is calculated in ',t_Gamma_end-t_Gamma_0,' seconds')

#-----------------------------------------------
#--(End) Main block 1: Initialise circulation---
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#---(Start) Main block 2: Iterate circulation---
#-----------------------------------------------
    Vortex_Wake_prev = Vortex_Wake
    #Calculate induced velocity
    i_iter = 0
    Error_old = 1
    while i_iter < n_iterations:
        Uind = factor*Ind_Vel_Mat_u*Gamma
        Vind = factor*Ind_Vel_Mat_v*Gamma
        Wind = factor*Ind_Vel_Mat_w*Gamma
        
        #Update vortex rings with new a_w
        if Double_rotor:
            a_w_weights = np.sin(np.pi/Ncp*np.arange(Ncp)).reshape((Ncp,1))
            a_w = float(np.average(Uind[:Ncp],axis=0,weights=a_w_weights)/V.U0)
            Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL, double=True, S_sep=L_sep, phase_dif=phase_dif)
        else:
            a_w_weights = np.sin(np.pi/Ncp*np.arange(Ncp)).reshape((Ncp,1))
            a_w = float(np.average(Uind[:Ncp],axis=0,weights=a_w_weights)/V.U0)
            Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL)
        
        if Double_rotor:
            [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, Gamma_1, double=True, phase_dif=phase_dif, d_sep=L_sep)
        else:
            [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, Gamma_1)
            
        #Determine new circulation
        if Double_rotor:
            GammaNew = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp, double=True, phase_dif=phase_dif)
        else:
            GammaNew = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp)
    
        #Calculate error
        referror = max(abs(GammaNew))
        referror = max(referror,0.001)
        Error = max(abs(GammaNew-Gamma))
        Error = Error/referror   
        
        #Update the circulation
        Gamma = GammaNew
        
        if Error < Error_margin:
            print('Error is ', np.round(float(Error),5))
            break
    
        #Determine if gamma is converging
        if Error < Error_old:
            print('Error is ', np.round(float(Error),5), 'and Gamma is converging with: ',np.round(float(Error-Error_old),5))
        else:
            print('Error is ', np.round(float(Error),5), 'and Gamma is diverging with: +',np.round(float(Error-Error_old),5))
            
        Error_old = Error
        print(i_iter)
        i_iter += 1
    

#-----------------------------------------------
#----(End) Main block 2: Iterate circulation----
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#-----(Start) Main block 3: Calculate Loads-----
#-----------------------------------------------
    
    #One simple function to obtain [fnorm, ftan, AoA, AngleInflow]
    if Double_rotor:
        results = LBE(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Twist_all_cp, polar_alpha, polar_cl, polar_cd, chord_all_cp, Vortex_Wake, double=True, phase_dif=phase_dif)
    else:
        results = LBE(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Twist_all_cp, polar_alpha, polar_cl, polar_cd, chord_all_cp, Vortex_Wake)
    
#-----------------------------------------------
#------(End) Main block 3: Calculate Loads------
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#---(Start) Post: Process results and display---
#-----------------------------------------------
    
    #Average results if double rotor configuration is used
    if Double_rotor:
        Fnorm = results[0]
        Ftan = results[1]
        alpha = results[2]
        inflow = results[3]
        Vax = results[4]
        Vtan = results[5]   
        Gamma_blade = Gamma
               
    else:
        Fnorm = results[0][:Ncp]
        Ftan = results[1][:Ncp]
        alpha = results[2][:Ncp] 
        inflow = results[3][:Ncp]
        Vax = results[4][:Ncp]
        Vtan = results[5][:Ncp]   
        Gamma_blade = Gamma[:Ncp]
    
    if not Double_rotor: 
        #Calculate thrust and power coefficient
        dr = (R_disL[1:]-R_disL[:-1])*V.R
        CT = np.sum(dr*np.array(Fnorm).reshape((Ncp,))*V.Nblades/(0.5*V.U0**2*V.rho*np.pi*V.R**2))
        CP = np.sum(dr*np.array(Ftan).reshape((Ncp,))*np.array(controlpoints).reshape((Ncp,))*V.R*V.Nblades*Omega/(0.5*V.U0**3*V.rho*np.pi*V.R**2))
        print('LL CT = ', np.round(CT,4))
        print('LL CP = ', np.round(CP,4))
    
        if Plot:
            #Axial induction factor versus radius  
            if case == 'turbine':
                a_i = 1 - Vax/V.U0
                aline = Vtan.reshape((Ncp,1))/(Omega*controlpoints*V.R) - 1
            else:
                a_i = Vax/V.U0 - 1
                aline = 1 - Vtan.reshape((Ncp,1))/(Omega*controlpoints*V.R)            
            fig_a = plt.figure(figsize=(12,6))
            ax_a = plt.gca()
            plt.title('Axial induction factor')
            plt.plot(controlpoints, a_i, 'k-x', label=r'$a$ - LL')
            plt.plot(controlpoints, aline, 'k--x', label=r'$a^,$')
            plt.grid()
            plt.xlabel('r/R')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            #Normal and tangential forces
            fig_force = plt.figure(figsize=(12,6))
            ax_force = plt.gca()
            plt.title(r'Normal and tangential force, non-dimensionalised by $\frac{1}{2} \rho U_\infty^2 R$')
            plt.plot(controlpoints, Fnorm/(0.5*V.U0**2*V.rho*V.R), 'k-x', label=r'Fnorm - LL')
            plt.plot(controlpoints, Ftan/(0.5*V.U0**2*V.rho*V.R), 'k--x', label=r'Ftan - LL')
            plt.grid()
            plt.xlabel('r/R')
            plt.legend()
            plt.tight_layout()
            plt.show()
                
            #Circulation distribution
            fig_circ = plt.figure(figsize=(12,6))
            ax_circ = plt.gca()
            plt.title(r'Circulation distribution, non-dimensionalised by $\frac{\pi U_\infty^2}{\Omega * NBlades}$')
            plt.plot(controlpoints, Gamma_blade/(np.pi*V.U0**2/(V.Nblades*Omega)), 'k-x', label=r'$\Gamma$ - LL')
            plt.grid()
            plt.xlabel('r/R')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            #Inflow distribution
            fig_inflow = plt.figure(figsize=(12,6))
            ax_inflow = plt.gca()
            plt.title('Angle distribution')
            plt.plot(controlpoints, inflow*180/np.pi, 'k--x', label='Inflowangle - LL')
            plt.plot(controlpoints, Twist_cp, 'g-s', label='Twist - LL')
            plt.plot(controlpoints, alpha, 'k-x', label=r'$\alpha$ - LL')
            plt.grid()
            plt.ylabel('(deg)')
            plt.xlabel('r/R')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
    else:
        dr = np.array(V.Nblades*list((R_disL[1:]-R_disL[:-1])*V.R))
        CT = np.sum(dr*np.array(Fnorm[:V.Nblades*Ncp]).reshape((V.Nblades*Ncp,))/(0.5*V.U0**2*V.rho*np.pi*V.R**2))
        CP = np.sum(dr*np.array(Ftan[:V.Nblades*Ncp]).reshape((V.Nblades*Ncp,))*np.array(controlpoints_all[:V.Nblades*Ncp]).reshape((V.Nblades*Ncp,))*V.R*Omega/(0.5*V.U0**3*V.rho*np.pi*V.R**2))
        print('LL two-rotor CT = ', np.round(CT,4))
        print('LL two-rotor CP = ', np.round(CP,4))
            
#-----------------------------------------------
#----(End) Post: Process results and display----
#-----------------------------------------------
###############################################################################
#-----------------------------------------------
#-------(Start) Sensitivity to wakelength-------
#-----------------------------------------------
#Repeat similar steps for sensitivity plots as was done in main program
else:
    for i in range(len(n_r_list)):
        n_t = n_t_lst[i]
        #Setup initial vortex wake structure
        t_VW_0 = time.time()
        Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w_init, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL)
        t_VW_end = time.time()
        print('Vortex wake geometry is determined in ', t_VW_end-t_VW_0,' seconds')
        
        #Setup Biot-Savart induction matrix for Gamma = 1
        [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, 1)
        t_ind_end = time.time()
        print('Induced velocity matrices are calculated in ',t_ind_end-t_VW_end,' seconds')
        
        #Initial estimate for circulation (with zero induced velocity)
        t_Gamma_0 = time.time()
        Uind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
        Vind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
        Wind = np.mat(np.zeros((Ncp*V.Nblades*n_rotors,1)))
        Gamma = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp)
        t_Gamma_end = time.time()
        print('Gamma is calculated in ',t_Gamma_end-t_Gamma_0,' seconds')
    
        #Calculate induced velocity
        i_iter = 0
        Error_old = 1
        while i_iter < n_iterations:
            Uind = Ind_Vel_Mat_u*Gamma
            Vind = Ind_Vel_Mat_v*Gamma
            Wind = Ind_Vel_Mat_w*Gamma
            
            #Update vortex rings with new a_w
            a_w_weights = np.sin(np.pi/Ncp*np.arange(Ncp)).reshape((Ncp,1))
            a_w = float(np.average(Uind[:Ncp],axis=0,weights=a_w_weights)/V.U0)
            Vortex_Wake = WG(case, V.U0, Omega, n_t, n_r, a_w, V.Nblades, R_disL*V.R, Chord_disL, Twist_disL)
            
            [Ind_Vel_Mat_u, Ind_Vel_Mat_v, Ind_Vel_Mat_w] = IV(delta, Vortex_Wake, controlpoints_all*V.R, 1)
                
            #Determine new circulation
            GammaNew = GF(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Vortex_Wake, Twist_all_cp, polar_alpha, polar_cl, chord_all_cp)
        
            #Calculate error
            referror = max(abs(GammaNew))
            referror = max(referror,0.001)
            Error = max(abs(GammaNew-Gamma))
            Error = Error/referror   
            
            #Update the circulation
            Gamma = GammaNew
            
            if Error < Error_margin:
                print('Error is ', np.round(float(Error),5))
                break

            Error_old = Error
            print(i_iter)
            i_iter += 1
      
        results = LBE(case, V.rho, V.U0, Uind, Vind, Wind, Omega, controlpoints*V.R, Twist_all_cp, polar_alpha, polar_cl, polar_cd, chord_all_cp, Vortex_Wake)
                
        Fnorm = results[0][:Ncp]
        Ftan = results[1][:Ncp]
        alpha = results[2][:Ncp] 
        inflow = results[3][:Ncp]
        Vax = results[4][:Ncp]
        Vtan = results[5][:Ncp]  
        Gamma_blade = Gamma[:Ncp]
        
        #Calculate thrust and power coefficient
        dr = (R_disL[1:]-R_disL[:-1])*V.R
        CT_sens[i] = np.sum(dr*np.array(Fnorm).reshape((Ncp,))*V.Nblades/(0.5*V.U0**2*V.rho*np.pi*V.R**2))
        CP_sens[i] = np.sum(dr*np.array(Ftan).reshape((Ncp,))*np.array(controlpoints).reshape((Ncp,))*V.R*V.Nblades*Omega/(0.5*V.U0**3*V.rho*np.pi*V.R**2))
        Gamma_sens[i,:] = Gamma_blade.reshape((Ncp,))
        Fnorm_sens[i,:] = Fnorm.reshape((Ncp,))
        Ftan_sens[i,:] = Ftan.reshape((Ncp,))

    #Plot results
    fig_force_sens = plt.figure(figsize=(12,6))
    plt.title('Force distributions for different wake lenghts')
    ax_force_sens = plt.gca()
    plt.plot(controlpoints,Fnorm_sens[0,:]/(0.5*V.U0**2*V.rho*V.R),'k'+lines[0], label=r'Fnorm, n_{r} = '+str(n_r_list[0]))
    plt.plot(controlpoints,Ftan_sens[0,:]/(0.5*V.U0**2*V.rho*V.R),'g'+lines[0], label=r'Ftan, n_{r} = '+str(n_r_list[0]))
    plt.legend()
    plt.xlabel('r/R')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    fig_gamma_sens = plt.figure(figsize=(12,6))
    plt.title('Circulation distributions for different wake lenghts')
    ax_gamma_sens = plt.gca()
    plt.plot(controlpoints,Gamma_sens[0,:]/(np.pi*V.U0**2/(V.Nblades*Omega)),'k'+lines[0], label=r'Fnorm, n_{r} = '+str(n_r_list[0]))
    plt.legend()
    plt.xlabel('r/R')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    fig_coef_sens = plt.figure(figsize=(12,6))
    plt.title('Thrust and power coefficient error for different wake lenghts')
    plt.plot(n_r_list[:-1],abs(CT_sens[:-1]-CT_sens[-1]),'k-o', label=r'$C_{t}$')
    plt.plot(n_r_list[:-1],abs(CP_sens[:-1]-CP_sens[-1]),'r-o', label=r'$C_{p}$')
    plt.grid()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('# of wake rotations')
    plt.tight_layout()
    plt.show()               
    for i in range(1,len(n_r_list)-1):
        ax_force_sens.plot(controlpoints,Fnorm_sens[i,:]/(0.5*V.U0**2*V.rho*V.R),'k'+lines[i], label=r'Fnorm, n_{r} = '+str(n_r_list[i]))
        ax_force_sens.plot(controlpoints,Ftan_sens[i,:]/(0.5*V.U0**2*V.rho*V.R),'g'+lines[i], label=r'Ftan, n_{r} = '+str(n_r_list[i]))
        ax_force_sens.legend()
        
        ax_gamma_sens.plot(controlpoints,Gamma_sens[i,:]/(np.pi*V.U0**2/(V.Nblades*Omega)),'k'+lines[i], label=r'\Gamma, n_{r} = '+str(n_r_list[i]))
        ax_gamma_sens.legend()   
#-----------------------------------------------
#--------(End) Sensitivity to wakelength--------
#-----------------------------------------------
t_end = time.time()
print('Total elapsed time is ',t_end-t_start,' seconds')