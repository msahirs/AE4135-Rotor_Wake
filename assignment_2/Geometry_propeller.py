#!/usr/bin/env python
# coding: utf-8

# geometry for propeller
# airfoil is ARA-D8% (polars in attachment on brightspace)
import numpy as np
R = 0.7  # in meters 
Nblades = 6  # number of blades
alt = 2000  # ISA altitude in meters
rho = 1.007 # density in kg/m^3
rootradius_R = 0.25  # blade starts at this, before that hub
tipradius_R = 1 # blade ends at this, end of the tip
Pblade = -75 # blade pitch in degrees (46 @ r/R=0.7)
Iangle = 0  # incidence angle in degrees
U0 = 60  # in m/s
rpm = 1200 # rev/min
CT_opt = 0.12
TSR = rpm/(2*np.pi) * R/U0 * rpm