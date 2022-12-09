# ************************************************************************ *
# *      This code solves for the viscous flow in a lid-driven cavity      *
# *************************************************************************
import numpy as np
import math
import copy
import os.path
from os import path
import re
# --- Variables for file handling ---
# --- All files are globally accessible ---

global fp1  # For output of iterative residual history
global fp2  # For output of field data (solution)
#   global fp3 # For writing the restart file
#   global fp4 # For reading the restart file
#   global fp5 # For output of final DE norms (only for MMS)
#   global fp6 # For debug: Uncomment for debugging.

global imax, jmax, neq, nmax
global zero, tenth, sixth, fifth, fourth, third, half, one, two, three, four, six
global iterout, imms, isgs, irstr, ipgorder, lim, cfl, Cx, Cy, toler, rkappa, Re, pinf, uinf, rho, rhoinv, xmin, xmax, ymin, ymax, Cx2, Cy2, fsmall
global rlength, rmu, vel2ref, dx, dy, rpi, phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy

# **Use these variables cautiously as these are globally accessible from all functions.**

global u         # Solution vector [p, u, v]^T at each node
global uold      # Previous (old) solution vector
global s        # Source term
global dt        # Local time step at each node
global artviscx  # Artificial viscosity in x-direction
global artviscy  # Artificial viscosity in y-direction
global ummsArray  # Array of umms values (funtion umms evaluated at all nodes)


# ************* Following are fixed parameters for array sizes **************
imax = 65     # * Number of points in the x-direction (use odd numbers only)
jmax = 65   # * Number of points in the y-direction (use odd numbers only)
neq = 3      # * Number of equation to be solved ( = 3: mass, x-mtm, y-mtm)

# **********************************************
# ****** All Global variables declared here. ***
# ***** These variables SHOULD not be changed **
# ********** by the program once set. **********
# **********************************************
# ***** The variables declared "const" CAN *****
# *** not be changed by the program once set ***
# **********************************************

# --------- Numerical Constants --------
zero = 0.0
tenth = 0.1
sixth = 1.0/6.0
fifth = 0.2
fourth = 0.25
third = 1.0/3.0
half = 0.5
one = 1.0
two = 2.0
three = 3.0
four = 4.0
six = 6.0

# -------- User sets inputs here  --------

nmax = 500000             # Maximum number of iterations
iterout = 500            # Number of time steps between solution output
imms = 1                  # Manufactured solution flag: = 1 for manuf. sol., = 0 otherwise
isgs = 0                  # Symmetric Gauss-Seidel  flag: = 1 for SGS, = 0 for point Jacobi
# Restart flag: = 1 for restart (file 'restart.in', = 0 for initial run
irstr = 0
# Order of pressure gradient: 0 = 2nd, 1 = 3rd (not needed)
ipgorder = 0
# variable to be used as the limiter sensor (= 0 for pressure)
lim = 0
residualOut = 10          # Number of timesteps between residual output

cfl = 0.5               # CFL number used to determine time step
Cx = 0.01               # Parameter for 4th order artificial viscosity in x
Cy = 0.01               # Parameter for 4th order artificial viscosity in y
toler = 1.e-10          # Tolerance for iterative residual convergence
rkappa = 0.1            # Time derivative preconditioning constant
Re = 100.0              # Reynolds number = rho*Uinf*L/rmu
# Initial pressure (N/m^2) -> from MMS value at cavity center
pinf = 0.801333844662
uinf = 1.0              # Lid velocity (m/s)
rho = 1.0               # Density (kg/m^3)
xmin = 0.0              # Cavity dimensions...: minimum x location (m)
xmax = 0.05             # maximum x location (m)
ymin = 0.0              # maximum y location (m)
ymax = 0.05             # maximum y location (m)
Cx2 = 0.0               # Coefficient for 2nd order damping (not required)
Cy2 = 0.0               # Coefficient for 2nd order damping (not required)
fsmall = 1e-20          # small parameter

# -- Derived input quantities (set by function 'set_derived_inputs' called from main)----

rhoinv = -99.9 	# Inverse density, 1/rho (m^3/kg)
rlength = -99.9 	# Characteristic length (m) [cavity width]
rmu = -99.9  		# Viscosity (N*s/m^2)
vel2ref = -99.9  	# Reference velocity squared (m^2/s^2)
dx = -99.9 		# Delta x (m)
dy = -99.9  		# Delta y (m)
rpi = -99.9 		# Pi = 3.14159... (defined below)

# -- ants for manufactured solutions ----
phi0 = [0.25, 0.3, 0.2]            # MMS ant
phix = [0.5, 0.15, 1.0/6.0]        # MMS amplitude ant
phiy = [0.4, 0.2, 0.25]            # MMS amplitude ant
phixy = [1.0/3.0, 0.25, 0.1]       # MMS amplitude ant
apx = [0.5, 1.0/3.0, 7.0/17.0] 	# MMS frequency ant
apy = [0.2, 0.25, 1.0/6.0]         # MMS frequency ant
apxy = [2.0/7.0, 0.4, 1.0/3.0]     # MMS frequency ant
fsinx = [0.0, 1.0, 0.0]            # MMS ant to determine sine vs. cosine
fsiny = [1.0, 0.0, 0.0]            # MMS ant to determine sine vs. cosine
fsinxy = [1.0, 1.0, 0.0]           # MMS ant to determine sine vs. cosine
# Note: fsin = 1 means the sine function
# Note: fsin = 0 means the cosine function
# Note: arrays here refer to the 3 variables


# **************************************************************************/
# *      					All Other	Functions					      */
# **************************************************************************/

# **************************************************************************


def set_derived_inputs():
    global imax, jmax
    global one
    global Re, uinf, rho, rhoinv, xmin, xmax, ymin, ymax
    global rlength, rmu, vel2ref, dx, dy, rpi

    # Inverse density, 1/rho (m^3/kg)
    rhoinv = one/rho
    # Characteristic length (m) [cavity width]
    rlength = xmax - xmin
    rmu = rho*uinf*rlength/Re                   # Viscosity (N*s/m^2) */
    # Reference velocity squared (m^2/s^2)
    vel2ref = uinf*uinf
    dx = (xmax - xmin)/(imax - 1)          # Delta x (m)
    dy = (ymax - ymin)/(jmax - 1)          # Delta y (m)
    rpi = math.acos(-one)                            # Pi = 3.14159...
    print("rho,V,L,mu,Re:\n"+str(rho)+" "+str(uinf) +
          " "+str(rlength)+" "+str(rmu)+" "+str(Re))

# ************************************************************************


def output_file_headers():

    # Uses global variable(s): imms, fp1, fp2
    # Note: The vector of primitive variables is:
    #               u = [p, u, v]^T
    # Set up output files (history and solution)

    global imms, fp1, fp2

    fp1 = open("history.dat", "w")
    fp1.write("TITLE = 'Cavity Iterative Residual History'\n")
    fp1.write("variables='Iteration''Time(s)''Res1''Res2''Res3'\n")

    fp2 = open("cavity.dat", "w")
    fp2.write("TITLE = 'Cavity Field Data'\n")
    if imms == 1:

        fp2.write("variables='x(m)''y(m)''p(N/m^2)''u(m/s)''v(m/s)'")
        fp2.write("'p-exact''u-exact''v-exact''DE-p''DE-u''DE-v'\n")

    else:

        if imms == 0:

            fp2.write("variables='x(m)''y(m)''p(N/m^2)''u(m/s)''v(m/s)\n")

        else:

            fp2.write("ERROR! imms must equal 0 or 1!!!\n")

    # Header for Screen Output
    print("Iter. Time (s)   dt (s)      Continuity    x-Momentum    y-Momentum\n")

# ************************************************************************


def initial(ninit, rtime, resinit):
    #
    # Uses global variable(s): zero, one, irstr, imax, jmax, neq, uinf, pinf
    # To modify: ninit, rtime, resinit, u, s

    # i                        # i index (x direction)
    # j                        # j index (y direction)
    # k                        # k index (# of equations)
    # x        # Temporary variable for x location
    # y        # Temporary variable for y location

    # This subroutine sets inital conditions in the cavity
    # Note: The vector of primitive variables is:
    #              u = (p, u, v)^T

    global zero, one, irstr, imax, jmax, neq, uinf, pinf, xmax, xmin, ymax, ymin
    global u, s, ummsArray

    if irstr == 0:   # Starting run from scratch
        ninit = 1          # set initial iteration to one
        rtime = 0.0       # set initial time to zero

        resinit[0:neq] = 1.0

        u[:, :, 0] = pinf*np.ones((imax, jmax))
        u[:, :, 1] = np.zeros((imax, jmax))
        u[:, :, 2] = np.zeros((imax, jmax))
        s = np.zeros_like(u)

        # Initialize lid (top) to freestream velocity
        u[:, jmax-1, 1] = uinf*np.ones(imax)

    elif irstr == 1:  # Restarting from previous run (file 'restart.in')
        if path.exists("./restart.out") == False:  # Note: 'restart.out' must exist!
            print('Error opening restart file. Stopping.\n')
            return
        # Remove any brackets from the data file
        with open("restart.out", 'r') as f:
            text = f.read()
            pattern = re.sub(r"[\([{})\]]", "", text)
        with open("restart.out", 'w') as newout_file:
            newout_file.write(pattern)

        # Read data from file
        with open("restart.out", 'r') as f:
            lines_list = f.readlines()
            # Need to known current iteration # and time value (first line)
            ninit, rtime = [float(x) for x in lines_list[0].split()]
            # Needs initial iterative residuals for scaling (second line)
            resinit[0], resinit[1], resinit[2] = [
                float(x) for x in lines_list[1].split()]
            list = [[float(x) for x in line.split()]
                    for line in lines_list[2:]]

        array = np.array(list)
        uproc = array[:, 2:5]
        for j in np.arange(0, jmax-1, 1):
            u[0:imax, j, 0] = uproc[j*imax:(j+1)*imax, 0]
            u[0:imax, j, 1] = uproc[j*imax:(j+1)*imax, 1]
            u[0:imax, j, 2] = uproc[j*imax:(j+1)*imax, 2]

        ninit = ninit + 1
        print("Restarting at iteration "+str(ninit)+"\n")
    else:
        print('ERROR: irstr must equal 0 or 1!\n')
        return

    # Initialize the ummsArray with values computed with umms function
    for j in np.arange(0, jmax, 1):
        for i in np.arange(0, imax, 1):
            for k in np.arange(0, neq, 1):
                x = (xmax - xmin)*(i)/(imax - 1)
                y = (ymax - ymin)*(j)/(jmax - 1)
                ummsArray[i, j, k] = umms(x, y, k)

    return ninit, rtime, resinit


# ************************************************************************


def set_boundary_conditions():
    #
    # Uses global variable(s): imms
    # To modify: u (via other functions: bndry() and bndrymms())

    #global imms

    # This subroutine determines the appropriate BC routines to call
    if (imms == 0):
        bndry()
    else:
        if (imms == 1):
            bndrymms()
        else:
            print("ERROR: imms must equal 0 or 1!\n")
            return

# ************************************************************************


def bndry():
    #
    # Uses global variable(s): zero, one, two, half, imax, jmax, uinf
    # To modify: u

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    global zero, two, half, imax, jmax, uinf
    global u

    # This applies the cavity boundary conditions

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

    # ----- Upper Wall BCs -----
    u[1:imax - 1, jmax - 1, 0] = 2*u[1:imax - 1, jmax - 2, 0] - u[1:imax - 1, jmax - 3, 0] # upper wall pressure
    u[0, jmax - 1, 0] = 2*u[1, jmax - 1, 0] - u[2, jmax - 1, 0] # uppper left corner point pressure
    u[imax - 1, jmax - 1, 0] = 2*u[imax - 2, jmax - 1, 0] - u[imax - 3, jmax - 1, 0] # upper right corner point pressure
    u[:, jmax - 1, 1] = uinf # upper wall u
    u[:, jmax - 1, 2] = 0 # upper wall v
    # ----- Left Wall BCs -----
    u[0, 1:jmax - 1, 0] = 2*u[1, 1:jmax - 1, 0] - u[2, 1:jmax - 1, 0] # left wall pressure
    u[0, 0, 0] = 2*u[0, 1, 0] - u[0, 2, 0] # lower left corner point pressure
    u[0, :, 1] = 0; # left wall u
    u[0, :, 2] = 0; # left wall v
    # ----- Bottom Wall BCs -----
    u[1:imax - 1, 0, 0] = 2*u[1:imax - 1, 1, 0] - u[1:imax - 1, 2, 0] # bottom wall pressure
    u[imax - 1, 0, 0] = 2*u[imax - 1, 1, 0] - u[imax - 1, 2, 0] # lower right corner point pressure
    u[:, 0, 1] = 0; # bottom wall u
    u[:, 0, 2] = 0; # bottom wall v
    # ----- Right Wall BCs -----
    u[imax - 1, 1:jmax - 1, 0] = 2*u[imax - 2, 1:jmax - 1, 0] - u[imax - 3, 1:jmax - 1, 0] # bottom wall pressure
    u[imax - 1, :, 1] = 0 # right wall u
    u[imax - 1, :, 2] = 0 # right wall v

# ************************************************************************


def bndrymms():
    #
    # Uses global variable(s): two, imax, jmax, neq, xmax, xmin, ymax, ymin, rlength
    # To modify: u
    # i                        # i index (x direction)
    # j                        # j index (y direction)
    # k                        # k index (# of equations)
    # x        # Temporary variable for x location
    # y        # Temporary variable for y location
    # This applies the cavity boundary conditions for the manufactured solution

    global two, imax, jmax, neq
    global u, ummsArray

    # Side Walls
    # Left Wall
    u[0, 1:jmax-1, 0:neq] = copy.deepcopy(ummsArray[0, 1:jmax-1, 0:neq])
    u[0, 1:jmax-1, 0] = 2*u[1, 1:jmax-1, 0]-u[2, 1:jmax-1, 0]  # 2nd Order BC
    # Right Wall
    u[imax-1, 1:jmax-1,
        0:neq] = copy.deepcopy(ummsArray[imax-1, 1:jmax-1, 0:neq])
    u[imax-1, 1:jmax-1, 0] = 2*u[imax-2, 1:jmax-1, 0] - \
        u[imax-3, 1:jmax-1, 0]  # 2nd Order BC

    # Top/Bottom Walls
    # Bottom Wall
    u[0:imax, 0, 0:neq] = copy.deepcopy(ummsArray[0:imax, 0, 0:neq])
    u[0:imax, 0, 0] = 2*u[0:imax, 1, 0] - u[0:imax, 2, 0]  # 2nd Order BC
    # Top Wall
    u[0:imax, jmax-1, 0:neq] = copy.deepcopy(ummsArray[0:imax, jmax-1, 0:neq])
    u[0:imax, jmax-1, 0] = 2*u[0:imax, jmax-2, 0] - \
        u[0:imax, jmax-3, 0]  # 2nd Order BC

# ************************************************************************


def umms(x, y, k):

    # Uses global variable(s): one, rpi, rlength
    # Inputs: x, y, k
    # To modify: <none>
    #Returns: umms

    # ummstmp; # Define return value for umms as % precision

    # termx       # Temp variable
    # termy       # Temp variable
    # termxy      # Temp variable
    # argx        # Temp variable
    # argy        # Temp variable
    # argxy       # Temp variable

    # This function returns the MMS exact solution

    global one, rpi, rlength
    global phi0, phix, phiy, phixy, apx, apy, apxy, fsinx, fsiny, fsinxy

    argx = apx[k]*rpi*x/rlength
    argy = apy[k]*rpi*y/rlength
    argxy = apxy[k]*rpi*x*y/rlength/rlength
    termx = phix[k]*(fsinx[k]*math.sin(argx)+(one-fsinx[k])*math.cos(argx))
    termy = phiy[k]*(fsiny[k]*math.sin(argy)+(one-fsiny[k])*math.cos(argy))
    termxy = phixy[k]*(fsinxy[k]*math.sin(argxy) +
                       (one-fsinxy[k])*math.cos(argxy))

    ummstmp = phi0[k] + termx + termy + termxy

    return ummstmp

# ************************************************************************


def write_output(n, resinit, rtime):

    # Uses global variable(s): imax, jmax, new, xmax, xmin, ymax, ymin, rlength, imms
    # Uses global variable(s): ninit, u, dt, resinit, rtime
    # To modify: <none>
    # Writes output and restart files.

    # i                        # i index (x direction)
    # j                        # j index (y direction)
    # k                        # k index (# of equations)

    # x        # Temporary variable for x location
    # y        # Temporary variable for y location

    global imax, jmax, xmax, xmin, ymax, ymin, imms
    global u, ummsArray
    global fp2, fp3

    # Field output
    fp2.write("zone T='n= "+str(n)+" ' "+"\n")
    fp2.write("I= "+str(imax)+" J= "+str(jmax)+"\n")
    fp2.write("DATAPACKING=POINT\n")

    if imms == 1:
        for i in np.arange(0, imax, 1):
            for j in np.arange(0, jmax, 1):
                x = (xmax - xmin)*i/(imax - 1)
                y = (ymax - ymin)*j/(jmax - 1)
                fp2.write(str(x)+" "+str(y)+" "+str(u[i, j, 0])+" "+str(u[i, j, 1])+" "+str(u[i, j, 2])+" "+str(ummsArray[i, j, 0])+" "+str(ummsArray[i, j, 1])+" "+str(ummsArray[i, j, 2])+" " +
                          str((u[i, j, 0]-ummsArray[i, j, 0]))+" "+str((u[i, j, 1]-ummsArray[i, j, 1]))+" "+str((u[i, j, 2]-ummsArray[i, j, 2]))+"\n")
    else:
        if imms == 0:
            for j in np.arange(0, jmax, 1):
                for i in np.arange(0, imax, 1):
                    x = (xmax - xmin)*i/(imax - 1)
                    y = (ymax - ymin)*j/(jmax - 1)
                    fp2.write(str(
                        x)+" "+str(y)+" "+str(u[i, j, 0])+" "+str(u[i, j, 1])+" "+str(u[i, j, 2])+"\n")
        else:
            print("ERROR: imms must equal 0 or 1!\n")
            return

    # Restart file: overwrites every 'iterout' iteration
    fp3 = open("restart.out", 'w')
    fp3.write(str(n)+" "+str(rtime)+"\n")
    fp3.write(str(resinit[0])+" "+str(resinit[1])+" "+str(resinit[2])+"\n")
    for j in np.arange(0, jmax, 1):
        for i in np.arange(0, imax, 1):
            x = (xmax - xmin)*(i)/(imax-1 - 1)
            y = (ymax - ymin)*(j)/(jmax-1 - 1)
            fp3.write(str(x)+" "+str(y)+" " +
                      str(u[i, j, 0])+" "+str(u[i, j, 1])+" "+str(u[i, j, 2])+"\n")
    fp3.close()
# ************************************************************************


def compute_source_terms():

    # Uses global variable(s): imax, jmax, imms, rlength, xmax, xmin, ymax, ymin
    # To modify: s (source terms)

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # x        # Temporary variable for x location
    # y        # Temporary variable for y location

    # Evaluate Source Terms Once at Beginning (only %erior po%s; will be zero for standard cavity)

    global imax, jmax, imms, xmax, xmin, ymax, ymin
    global s

    for j in np.arange(1, jmax, 1):
        for i in np.arange(1, imax, 1):
            x = (xmax - xmin)*(i)/(imax - 1)
            y = (ymax - ymin)*(j)/(jmax - 1)
            s[i, j, 0] = (imms)*srcmms_mass(x, y)
            s[i, j, 1] = (imms)*srcmms_xmtm(x, y)
            s[i, j, 2] = (imms)*srcmms_ymtm(x, y)

# ************************************************************************


def srcmms_mass(x, y):

    # Uses global variable(s): rho, rpi, rlength
    # Inputs: x, y
    # To modify: <none>
    #Returns: srcmms_mass
    # srcmasstmp; # Define return value for srcmms_mass as % precision

    # dudx; 	# Temp variable: u velocity gradient in x direction
    # dvdy;  # Temp variable: v velocity gradient in y direction

    # This function returns the MMS mass source term

    #global rho, rpi, rlength
    #global phix, phiy, phixy, apx, apy, apxy

    dudx = phix[1]*apx[1]*rpi/rlength*math.cos(apx[1]*rpi*x/rlength) + phixy[1] * \
        apxy[1]*rpi*y/rlength/rlength*math.cos(apxy[1]*rpi*x*y/rlength/rlength)

    dvdy = -phiy[2]*apy[2]*rpi/rlength*math.sin(apy[2]*rpi*y/rlength) - phixy[2] * \
        apxy[2]*rpi*x/rlength/rlength*math.sin(apxy[2]*rpi*x*y/rlength/rlength)

    srcmasstmp = rho*dudx + rho*dvdy

    return srcmasstmp
# ************************************************************************


def srcmms_xmtm(x, y):

    # Uses global variable(s): rho, rpi, rmu, rlength
    # Inputs: x, y
    # To modify: <none>
    #Returns: srcmms_xmtm

    # srcxmtmtmp; # Define return value for srcmms_xmtm as % precision

    # dudx; 	# Temp variable: u velocity gradient in x direction
    # dudy;  # Temp variable: u velocity gradient in y direction
    # termx;        # Temp variable
    # termy;        # Temp variable
    # termxy;       # Temp variable
    # uvel;         # Temp variable: u velocity
    # vvel;         # Temp variable: v velocity
    # dpdx;         # Temp variable: pressure gradient in x direction
    # d2udx2;       # Temp variable: 2nd derivative of u velocity in x direction
    # d2udy2;       # Temp variable: 2nd derivative of u velocity in y direction

    # This function returns the MMS x-momentum source term

    global rho, rpi, rmu, rlength
    global phi0, phix, phiy, phixy, apx, apy, apxy

    termx = phix[1]*math.sin(apx[1]*rpi*x/rlength)
    termy = phiy[1]*math.cos(apy[1]*rpi*y/rlength)
    termxy = phixy[1]*math.sin(apxy[1]*rpi*x*y/rlength/rlength)
    uvel = phi0[1] + termx + termy + termxy

    termx = phix[2]*math.cos(apx[2]*rpi*x/rlength)
    termy = phiy[2]*math.cos(apy[2]*rpi*y/rlength)
    termxy = phixy[2]*math.cos(apxy[2]*rpi*x*y/rlength/rlength)
    vvel = phi0[2] + termx + termy + termxy

    dudx = phix[1]*apx[1]*rpi/rlength*math.cos(apx[1]*rpi*x/rlength) + phixy[1] * \
        apxy[1]*rpi*y/rlength/rlength*math.cos(apxy[1]*rpi*x*y/rlength/rlength)

    dudy = -phiy[1]*apy[1]*rpi/rlength*math.sin(apy[1]*rpi*y/rlength) + phixy[1] * \
        apxy[1]*rpi*x/rlength/rlength * \
        math.cos(apxy[1]*rpi*x*y/rlength/rlength)

    dpdx = -phix[0]*apx[0]*rpi/rlength*math.sin(apx[0]*rpi*x/rlength) + phixy[0] * \
        apxy[0]*rpi*y/rlength/rlength * \
        math.cos(apxy[0]*rpi*x*y/rlength/rlength)

    d2udx2 = -phix[1]*np.square(apx[1]*rpi/rlength) * math.sin(apx[1]*rpi*x/rlength) - phixy[1] * \
        np.square(apxy[1]*rpi*y/rlength/rlength) * \
        math.sin(apxy[1]*rpi*x*y/rlength/rlength)

    d2udy2 = -phiy[1]*np.square(apy[1]*rpi/rlength) * math.cos(apy[1]*rpi*y/rlength) - phixy[1] * \
        np.square(apxy[1]*rpi*x/rlength/rlength) * \
        math.sin(apxy[1]*rpi*x*y/rlength/rlength)

    srcxmtmtmp = rho*uvel*dudx + rho*vvel*dudy + dpdx - rmu*(d2udx2 + d2udy2)

    return srcxmtmtmp
# ************************************************************************


def srcmms_ymtm(x, y):

    # Uses global variable(s): rho, rpi, rmu, rlength
    # Inputs: x, y
    # To modify: <none>
    #Returns: srcmms_ymtm

    # srcymtmtmp; # Define return value for srcmms_ymtm as % precision

    # dvdx;         # Temp variable: v velocity gradient in x direction
    # dvdy;         # Temp variable: v velocity gradient in y direction
    # termx;        # Temp variable
    # termy;        # Temp variable
    # termxy;       # Temp variable
    # uvel;         # Temp variable: u velocity
    # vvel;         # Temp variable: v velocity
    # dpdy;         # Temp variable: pressure gradient in y direction
    # d2vdx2;       # Temp variable: 2nd derivative of v velocity in x direction
    # d2vdy2;       # Temp variable: 2nd derivative of v velocity in y direction

    # This function returns the MMS y-momentum source term

    global rho, rpi, rmu, rlength
    global phi0, phix, phiy, phixy, apx, apy, apxy

    termx = phix[1]*math.sin(apx[1]*rpi*x/rlength)
    termy = phiy[1]*math.cos(apy[1]*rpi*y/rlength)
    termxy = phixy[1]*math.sin(apxy[1]*rpi*x*y/rlength/rlength)
    uvel = phi0[1] + termx + termy + termxy

    termx = phix[2]*math.cos(apx[2]*rpi*x/rlength)
    termy = phiy[2]*math.cos(apy[2]*rpi*y/rlength)
    termxy = phixy[2]*math.cos(apxy[2]*rpi*x*y/rlength/rlength)
    vvel = phi0[2] + termx + termy + termxy

    dvdx = -phix[2]*apx[2]*rpi/rlength*math.sin(apx[2]*rpi*x/rlength) - phixy[2] * \
        apxy[2]*rpi*y/rlength/rlength * \
        math.sin(apxy[2]*rpi*x*y/rlength/rlength)

    dvdy = -phiy[2]*apy[2]*rpi/rlength*math.sin(apy[2]*rpi*y/rlength) - phixy[2] * \
        apxy[2]*rpi*x/rlength/rlength * \
        math.sin(apxy[2]*rpi*x*y/rlength/rlength)

    dpdy = phiy[0]*apy[0]*rpi/rlength*math.cos(apy[0]*rpi*y/rlength) + phixy[0] * \
        apxy[0]*rpi*x/rlength/rlength * \
        math.cos(apxy[0]*rpi*x*y/rlength/rlength)

    d2vdx2 = -phix[2]*np.square(apx[2]*rpi/rlength) * math.cos(apx[2]*rpi*x/rlength) - phixy[2] * \
        np.square(apxy[2]*rpi*y/rlength/rlength) * \
        math.cos(apxy[2]*rpi*x*y/rlength/rlength)

    d2vdy2 = -phiy[2]*np.square(apy[2]*rpi/rlength) * math.cos(apy[2]*rpi*y/rlength) - phixy[2] * \
        np.square(apxy[2]*rpi*x/rlength/rlength) * \
        math.cos(apxy[2]*rpi*x*y/rlength/rlength)

    srcymtmtmp = rho*uvel*dvdx + rho*vvel*dvdy + dpdy - rmu*(d2vdx2 + d2vdy2)

    return srcymtmtmp
# ************************************************************************


def compute_time_step(dtmin):

    # Uses global variable(s): one, two, four, half, fourth
    # Uses global variable(s): vel2ref, rmu, rho, dx, dy, cfl, rkappa, imax, jmax
    # Uses: u
    # To Modify: dt, dtmin

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # dtvisc       # Viscous time step stability criteria (constant over domain)
    # uvel2        # Local x velocity squared
    # vvel2        # Local y velocity squared, added variable
    # beta2        # Beta squared parameter for time derivative preconditioning
    # lambda_x     # Max absolute value eigenvalue in (x,t)
    # lambda_y     # Max absolute value eigenvalue in (y,t)
    # lambda_max   # Max absolute value eigenvalue (used in convective time step computation)
    # dtconv       # Local convective time step restriction

    global four, half, fourth
    global vel2ref, rmu, rho, dx, dy, cfl, rkappa, imax, jmax
    global u, dt

    # !**************************************************************
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************
    # !**************************************************************
    
    dtvisc = fourth*(dx*dy)/(rmu/rho)
    temp_dtvisc_array = np.zeros((imax - 2, jmax - 2))
    temp_dtvisc_array[:, :] = dtvisc
    uvel2 = u[1:imax - 1, 1:jmax - 1, 1]**2
    vvel2 = u[1:imax - 1, 1:jmax - 1, 2]**2
    vel2ref = uinf**2
    temp_rkappa_array = np.zeros((imax - 2, jmax - 2)) # added variable to compare arrays
    temp_rkappa_array[:, :] = vel2ref*rkappa
    temp_dx_array = np.zeros((imax - 2, jmax - 2)) # added variable to compare arrays
    temp_dx_array[:, :] = dx
    beta2 = np.maximum(uvel2 + vvel2, temp_rkappa_array)
    lambda_x = half*(np.abs(u[1:imax - 1, 1:jmax - 1, 1]) + np.sqrt(uvel2 + four*beta2))
    lambda_y = half*(np.abs(u[1:imax - 1, 1:jmax - 1, 2]) + np.sqrt(vvel2 + four*beta2))
    lambda_max = np.maximum(lambda_x, lambda_y)
    dtconv = np.divide(temp_dx_array, lambda_max)
    dtmin = np.zeros((imax, jmax))
    dtmin[1:imax - 1, 1:jmax - 1] = cfl*np.minimum(dtconv, dtvisc) 

    return dtmin

# ************************************************************************


def Compute_Artificial_Viscosity():

    # Uses global variable(s): zero, one, two, four, six, half, fourth
    # Uses global variable(s): imax, jmax, lim, rho, dx, dy, Cx, Cy, Cx2, Cy2, fsmall, vel2ref, rkappa
    # Uses: u
    # To Modify: artviscx, artviscy

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # uvel2        # Local x velocity squared
    # vvel2        # Local y velocity squared, added variable
    # beta2        # Beta squared parameter for time derivative preconditioning
    # lambda_x     # Max absolute value e-value in (x,t)
    # lambda_y     # Max absolute value e-value in (y,t)
    # d4pdx4       # 4th derivative of pressure w.r.t. x
    # d4pdy4       # 4th derivative of pressure w.r.t. y
    # # d2pdx2       # 2nd derivative of pressure w.r.t. x [these are not used]
    # # d2pdy2       # 2nd derivative of pressure w.r.t. y [these are not used]
    # # pfunct1      # Temporary variable for 2nd derivative damping [these are
    # not used]
    # # pfunct2      # Temporary variable for 2nd derivative damping [these are
    # not used]

    global two, four, six, half
    global imax, jmax, lim, rho, dx, dy, Cx, Cy, Cx2, Cy2, fsmall, vel2ref, rkappa
    global u
    global artviscx, artviscy

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    
    # Copied from time-step computaion function -----
    uvel2 = u[1:imax - 1, 1:jmax - 1, 1]**2
    vvel2 = u[1:imax - 1, 1:jmax - 1, 2]**2
    vel2ref = uinf**2
    temp_rkappa_array = np.zeros((imax - 2, jmax - 2)) # added variable to compare arrays
    temp_rkappa_array[:, :] = vel2ref*rkappa
    beta2 = np.maximum(uvel2 + vvel2, temp_rkappa_array)
    lambda_x = half*(np.abs(u[1:imax - 1, 1:jmax - 1, 1]) + np.sqrt(uvel2 + four*beta2))
    lambda_y = half*(np.abs(u[1:imax - 1, 1:jmax - 1, 2]) + np.sqrt(vvel2 + four*beta2))
    # -----
    # loop sets interior artviscs, points near boundary need separate treatment
    for j in range(2, jmax - 2, 1):
        for i in range(2, imax - 2, 1):
            d4pdx4 = (u[i + 2, j, 0] - u[i + 1, j, 0] + 6*u[i, j, 0] - 4*u[i - 1, j, 0] + u[i - 2, j, 0])/dx**4
            d4pdy4 = (u[i, j + 2, 0] - u[i, j + 1, 0] + 6*u[i, j, 0] - 4*u[i, j - 1, 0] + u[i, j - 2, 0])/dy**4
            artviscx[i, j] = -((lambda_x[i - 2, j - 2]*Cx*dx**3)/beta2[i,j])*d4pdx4
            artviscy[i, j] = -((lambda_y[i - 2, j - 2]*Cy*dy**3)/beta2[i,j])*d4pdy4
    # setting artviscs for points near boundary, extrapolated as with pressure boundary conditions
    #
    # ---- check if artvisc is needed at boundaries ----
    #
    # near left wall
    artviscx[1, 2:jmax - 2] = 2*artviscx[2, 2:jmax - 2] - artviscx[3, 2:jmax - 2]
    artviscy[1, 2:jmax - 2] = 2*artviscy[2, 2:jmax - 2] - artviscy[3, 2:jmax - 2]
    # near upper wall with upper left corner point
    artviscx[2:imax - 2, jmax - 2] = 2*artviscx[2:imax - 2, jmax - 3] - artviscx[2:imax - 2, jmax - 4]
    artviscy[2:imax - 2, jmax - 2] = 2*artviscy[2:imax - 2, jmax - 3] - artviscy[2:imax - 2, jmax - 4]
    artviscx[1, jmax - 2] = 2*artviscx[2, jmax - 2] - artviscx[3, jmax - 2]
    artviscy[1, jmax - 2] = 2*artviscy[2, jmax - 2] - artviscy[3, jmax - 2]
    # near right wall with upper right corner point
    artviscx[imax - 2, 2:jmax - 2] = 2*artviscx[imax - 3, 2:jmax - 2] - artviscx[imax - 4, 2:jmax - 2]
    artviscy[imax - 2, 2:jmax - 2] = 2*artviscy[imax - 3, 2:jmax - 2] - artviscy[imax - 4, 2:jmax - 2]
    artviscx[imax - 2, jmax - 2] = 2*artviscx[imax - 2, jmax - 3] - artviscx[imax - 2, jmax - 4]
    artviscy[imax - 2, jmax - 2] = 2*artviscy[imax - 2, jmax - 3] - artviscy[imax - 2, jmax - 4]
    # near lower wall with lower left and right corner points
    artviscx[2:imax - 2, 1] = 2*artviscx[2:imax - 2, 2] - artviscx[2:imax - 2, 3]
    artviscy[2:imax - 2, 1] = 2*artviscy[2:imax - 2, 2] - artviscy[2:imax - 2, 3]
    artviscx[imax - 2, 1] = 2*artviscx[imax - 2, 2] - artviscx[imax - 2, 3]
    artviscy[imax - 2, 1] = 2*artviscy[imax - 2, 2] - artviscy[imax - 2, 3]
    artviscx[1,1] = 2*artviscx[1, 2] - artviscx[1, 3]
    artviscy[1,1] = 2*artviscy[1, 2] - artviscy[1, 3]

# ************************************************************************


def SGS_forward_sweep():

    # Uses global variable(s): two, three, six, half
    # Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
    #                      xmax, xmin, ymax, ymin, rmu, vel2ref
    # Uses: artviscx, artviscy, dt, s
    # To Modify: u

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # dpdx         # First derivative of pressure w.r.t. x
    # dudx         # First derivative of x velocity w.r.t. x
    # dvdx         # First derivative of y velocity w.r.t. x
    # dpdy         # First derivative of pressure w.r.t. y
    # dudy         # First derivative of x velocity w.r.t. y
    # dvdy         # First derivative of y velocity w.r.t. y
    # d2udx2       # Second derivative of x velocity w.r.t. x
    # d2vdx2       # Second derivative of y velocity w.r.t. x
    # d2udy2       # Second derivative of x velocity w.r.t. y
    # d2vdy2       # Second derivative of y velocity w.r.t. y
    # beta2        # Beta squared parameter for time derivative preconditioning
    # uvel2        # Velocity squared

    global two, half
    global imax, jmax, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref
    global artviscx, artviscy, dt, s, u

    # Symmetric Gauss-Siedel: Forward Sweep

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */


# ************************************************************************


def SGS_backward_sweep():

    # Uses global variable(s): two, three, six, half
    # Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
    #                      xmax, xmin, ymax, ymin, rmu, vel2ref
    # Uses: artviscx, artviscy, dt, s
    # To Modify: u

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # dpdx         # First derivative of pressure w.r.t. x
    # dudx         # First derivative of x velocity w.r.t. x
    # dvdx         # First derivative of y velocity w.r.t. x
    # dpdy         # First derivative of pressure w.r.t. y
    # dudy         # First derivative of x velocity w.r.t. y
    # dvdy         # First derivative of y velocity w.r.t. y
    # d2udx2       # Second derivative of x velocity w.r.t. x
    # d2vdx2       # Second derivative of y velocity w.r.t. x
    # d2udy2       # Second derivative of x velocity w.r.t. y
    # d2vdy2       # Second derivative of y velocity w.r.t. y
    # beta2        # Beta squared parameter for time derivative preconditioning
    # uvel2        # Velocity squared

    global two, half
    global imax, jmax, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref
    global artviscx, artviscy, dt, s, u

    # Symmetric Gauss-Siedel: Backward Sweep

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */

# ************************************************************************


def point_Jacobi():

    # Uses global variable(s): two, three, six, half
    # Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
    #                      xmax, xmin, ymax, ymin, rmu, vel2ref
    # Uses: uold, artviscx, artviscy, dt, s
    # To Modify: u

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # dpdx         # First derivative of pressure w.r.t. x
    # dudx         # First derivative of x velocity w.r.t. x
    # dvdx         # First derivative of y velocity w.r.t. x
    # dpdy         # First derivative of pressure w.r.t. y
    # dudy         # First derivative of x velocity w.r.t. y
    # dvdy         # First derivative of y velocity w.r.t. y
    # d2udx2       # Second derivative of x velocity w.r.t. x
    # d2vdx2       # Second derivative of y velocity w.r.t. x
    # d2udy2       # Second derivative of x velocity w.r.t. y
    # d2vdy2       # Second derivative of y velocity w.r.t. y
    # beta2        # Beta squared parameter for time derivative preconditioning
    # uvel2        # x velocity squared
    # vvel2        # y velocity squared, added variable
    global two, half
    global imax, jmax, rho, rhoinv, dx, dy, rkappa, rmu, vel2ref
    global u, uold, artviscx, artviscy, dt, s

    # Point Jacobi method

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    
    # Copied from time-step computaion function -----
    uvel2 = uold[1:imax - 1, 1:jmax - 1, 1]**2
    vvel2 = uold[1:imax - 1, 1:jmax - 1, 2]**2
    vel2ref = uinf**2
    temp_rkappa_array = np.zeros((imax - 2, jmax - 2)) # added variable to compare arrays
    temp_rkappa_array[:, :] = vel2ref*rkappa
    beta2 = np.maximum(uvel2 + vvel2, temp_rkappa_array)
    # -----`
    # using local timestepping
    dt = dtmin
    # -----  
    for j in range(1, jmax - 1, 1):
        for i in range(1, imax - 1, 1):
            dpdx = (uold[i + 1, j, 0] - uold[i - 1, j, 0])/(2*dx)
            dudx = (uold[i + 1, j, 1] - uold[i - 1, j, 1])/(2*dx)
            dvdx = (uold[i + 1, j, 2] - uold[i - 1, j, 2])/(2*dx)
            dpdy = (uold[i, j + 1, 0] - uold[i, j - 1, 0])/(2*dy)
            dudy = (uold[i, j + 1, 1] - uold[i, j - 1, 1])/(2*dy)
            dvdy = (uold[i, j + 1, 2] - uold[i, j - 1, 2])/(2*dy)
            d2udx2 = (uold[i + 1, j, 1] - 2*uold[i, j, 1] + uold[i - 1, j, 1])/(dx**2)
            d2vdx2 = (uold[i + 1, j, 2] - 2*uold[i, j, 2] + uold[i - 1, j, 2])/(dx**2)
            d2udy2 = (uold[i, j + 1, 1] - 2*uold[i, j, 1] + uold[i, j - 1, 1])/(dy**2)
            d2vdy2 = (uold[i, j + 1, 2] - 2*uold[i, j, 2] + uold[i, j - 1, 2])/(dy**2)
            # ----- continuity equation -----
            u[i, j, 0] = uold[i, j, 0] - beta2[i - 1,j - 1]*dt[i, j]*(rho*(dudx + dvdy) - artviscx[i,j] - artviscy[i,j] - s[i, j, 0])
            # ----- x momentum equation -----
            u[i, j, 1] = uold[i, j, 1] - dt[i, j]*rhoinv*(rho*(uold[i, j, 1]*dudx + uold[i, j, 2]*dudy) + dpdx - rmu*(d2udx2 + d2udy2) - s[i, j, 1])
            # ----- y momentum equation -----
            u[i, j, 2] = uold[i, j, 2] - dt[i, j]*rhoinv*(rho*(uold[i, j, 1]*dvdx + uold[i, j, 2]*dvdy) + dpdy - rmu*(d2vdx2 + d2vdy2) - s[i, j, 2])
    
# ************************************************************************


def pressure_rescaling():

    # Uses global variable(s): imax, jmax, imms, xmax, xmin, ymax, ymin, rlength, pinf
    # To Modify: u

    # i                        # i index (x direction)
    # j                        # j index (y direction)

    # iref                      # i index location of pressure rescaling point
    # jref                      # j index location of pressure rescaling point

    # x        # Temporary variable for x location
    # y        # Temporary variable for y location
    # deltap   # delta_pressure for rescaling all values

    #global imax, jmax, imms, xmax, xmin, ymax, ymin, pinf
    #global u

    iref = int((imax-1)/2+1)     # Set reference pressure to center of cavity
    jref = int((jmax-1)/2+1)
    if imms == 1:
        x = (xmax - xmin)*(iref-1)/(imax - 1)
        y = (ymax - ymin)*(jref-1)/(jmax - 1)
        deltap = u[iref, jref, 0] - umms(x, y, 0)  # Constant in MMS
    else:
        deltap = u[iref, jref, 0] - pinf  # Reference pressure

    # j=np.arange(1,jmax,1)
    # i=np.arange(1,imax,1)
    u[:, :, 0] = u[:, :, 0] - deltap


# ************************************************************************


def check_iterative_convergence(n, res, resinit, ninit, rtime, dtmin):

    # Uses global variable(s): zero
    # Uses global variable(s): imax, jmax, neq, fsmall
    # Uses: n, u, uold, dt, res, resinit, ninit, rtime, dtmin
    # To modify: conv

    # i                        # i index (x direction)
    # j                        # j index (y direction)
    # k                        # k index (# of equations)

    global zero
    global imax, jmax, neq, fsmall
    global u, uold, dt, fp1

    # Compute iterative residuals to monitor iterative convergence

    # !************************************************************** */
    # !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
    # !************************************************************** */
    
    # using L2 norm
    
    r2 = np.zeros((imax - 2, jmax - 2, neq ))
    #init_norm = np.zeros((neq, 1))
    
    for i in range(1, imax - 1, 1):
        for j in range(1, jmax - 1, 1):
            # ----- code copied from point-jacobi -----
            dpdx = (uold[i + 1, j, 0] - uold[i - 1, j, 0])/(2*dx)
            dudx = (uold[i + 1, j, 1] - uold[i - 1, j, 1])/(2*dx)
            dvdx = (uold[i + 1, j, 2] - uold[i - 1, j, 2])/(2*dx)
            dpdy = (uold[i, j + 1, 0] - uold[i, j - 1, 0])/(2*dy)
            dudy = (uold[i, j + 1, 1] - uold[i, j - 1, 1])/(2*dy)
            dvdy = (uold[i, j + 1, 2] - uold[i, j - 1, 2])/(2*dy)
            d2udx2 = (uold[i + 1, j, 1] - 2*uold[i, j, 1] + uold[i - 1, j, 1])/dx**2
            d2vdx2 = (uold[i + 1, j, 2] - 2*uold[i, j, 2] + uold[i - 1, j, 2])/dx**2
            d2udy2 = (uold[i, j + 1, 1] - 2*uold[i, j, 1] + uold[i, j - 1, 1])/dy**2
            d2vdy2 = (uold[i, j + 1, 2] - 2*uold[i, j, 2] + uold[i, j - 1, 2])/dy**2
            # -----
            r2[i - 1, j - 1, 0] = rho*(dudx + dvdy) - artviscx[i, j] - artviscy[i, j] - s[i, j, 0]
            r2[i - 1, j - 1, 1] = rho*(uold[i, j, 1]*dudx + uold[i, j, 2]*dudy) + dpdx - rmu*(d2udx2 + d2udy2) - s[i, j, 1]
            r2[i - 1, j - 1, 2] = rho*(uold[i, j, 1]*dvdx + uold[i, j, 2]*dvdy) + dpdy - rmu*(d2vdx2 + d2vdy2) - s[i, j, 2] # ----- steady portion of discretization, copied from point-jacobi

    #if n == ninit:
     #   init_norm[0] = np.sqrt(np.sum(np.square(r2[:, :, 0]))/(imax*jmax))
      #  init_norm[1] = np.sqrt(np.sum(np.square(r2[:, :, 1]))/(imax*jmax))
       # init_norm[2] = np.sqrt(np.sum(np.square(r2[:, :, 2]))/(imax*jmax))
        
    res[0] = np.sqrt(np.sum(np.square(r2[:, :, 0]))/(imax*jmax))#/init_norm[0]
    res[1] = np.sqrt(np.sum(np.square(r2[:, :, 1]))/(imax*jmax))#/init_norm[1]
    res[2] = np.sqrt(np.sum(np.square(r2[:, :, 2]))/(imax*jmax))#/init_norm[2]
    
    conv = min(res[0], res[1], res[2])
    
    # Write iterative residuals every 10 iterations
    if n % 10 == 0 or n == ninit:
        fp1.write(str(n)+" "+str(rtime)+" " +
                  str(res[0])+" "+str(res[1])+" "+str(res[2]) + "\n")
        print(str(n)+" "+str(rtime)+" "+str(dtmin)+" " +
              str(res[0])+" "+str(res[1])+" "+str(res[2])+"\n")
        # Maybe a need to format this better

    # Write header for iterative residuals every 200 iterations
    if n % 200 == 0 or n == ninit:
        print("Iter."+" "+"Time (s)"+" "+"dt (s)"+" " +
              "Continuity"+" "+"x-Momentum"+" "+"y-Momentum\n")

    return res, resinit, conv


# ************************************************************************
def Discretization_Error_Norms(rL1norm, rL2norm, rLinfnorm):

    # Uses global variable(s): zero
    # Uses global variable(s): imax, jmax, neq, imms, xmax, xmin, ymax, ymin, rlength
    #Uses: u
    # To modify: rL1norm, rL2norm, rLinfnorm

    # i                        # i index (x direction)
    # j                        # j index (y direction)
    # k                        # k index (# of equations)

    # x        # Temporary variable for x location
    # y        # Temporary variable for y location
    # DE   	# Discretization error (absolute value)

    #global zero, imax, jmax, neq, imms, xmax, xmin, ymax, ymin, u

    if imms == 1:

        # !**************************************************************
        # !************ADD CODING HERE FOR INTRO CFD STUDENTS************
        #!***************************************************************

        return  # remove this once you add code

    # ***************************************************************************


# ************************************************************************
#      						Main Function
# ************************************************************************
# ----- Looping indices --------
i = 0                       # i index (x direction)
j = 0                       # j index (y direction)
k = 0                       # k index (# of equations)
n = 0	                   # Iteration number index

conv = -99.9  # Minimum of iterative residual norms from three equations


# --------- Solution variables declaration --------

ninit = 0        	# Initial iteration number (used for restart file)

#    u(imax,jmax,neq);          # Solution vector (p, u, v)^T at each node
#    uold(imax,jmax,neq);       # Previous (old) solution vector
#    s(imax,jmax,neq);          # Source term
#    dt(imax,jmax);             # Local time step at each node
#    artviscx(imax,jmax);       # Artificial viscosity in x-direction
#    artviscy(imax,jmax);       # Artificial viscosity in y-direction
res = [0, 0, 0]              # Iterative residual for each equation
# Initial iterative residual for each equation (from iteration 1)
resinit = [0, 0, 0]
# L1 norm of discretization error for each equation
rL1norm = [0, 0, 0]
# L2 norm of discretization error for each equation
rL2norm = [0, 0, 0]
# Linfinity norm of discretization error for each equation
rLinfnorm = [0, 0, 0]
rtime = -99.9         # Variable to estimate simulation time
# Minimum time step for a given iteration (initialized large)
dtmin = 1.0e99

x = -99.9       # Temporary variable for x location
y = -99.9       # Temporary variable for y location

# Solution variables initialization with dummy values
# for i=1:imax
#  for j=1:jmax
#      dt(i,j) = -99.9;
#      artviscx(i,j) = -99.9;
#      artviscy(i,j) = -99.9;
#    for k=1:neq
#      u(i,j,k) = -99.9;
#      uold(i,j,k) = -99.9;
#      s(i,j,k) = -99.9;
#      res(k) = -99.9;
#      resinit(k) = -99.9;
#      res(k) = -99.9;
#      rL1norm(k) = -99.9;
#      rL2norm(k) = -99.9;
#      rLinfnorm(k) = -99.9;


dt = np.zeros((imax, jmax))
artviscx = np.zeros((imax, jmax))
artviscy = np.zeros((imax, jmax))
u = np.zeros((imax, jmax, neq))
uold = np.zeros((imax, jmax, neq))
ummsArray = np.zeros((imax, jmax, neq))
s = np.zeros((imax, jmax, neq))
res = np.zeros((neq, 1))
resinit = np.zeros((neq, 1))
rL1norm = np.zeros((neq, 1))
rL2norm = np.zeros((neq, 1))
rLinfnorm = np.zeros((neq, 1))

np.place(dt, dt == 0, -99.9)
# np.place(artviscx, artviscx == 0, -99.9)
# np.place(artviscy, artviscy == 0, -99.9)
np.place(u, u == 0, -99.9)
np.place(uold, uold == 0, -99.9)
np.place(s, s == 0, -99.9)
np.place(res, res == 0, -99.9)
np.place(resinit, resinit == 0, -99.9)
np.place(rL1norm, rL1norm == 0, -99.9)
np.place(rL2norm, rL2norm == 0, -99.9)
np.place(rLinfnorm, rLinfnorm == 0, -99.9)


# Debug output: Uncomment and modify if debugging
# fp6 = fopen("./Debug.dat","w");
# print(fp6,"TITLE = \"Debug Data Data\"\n");
# print(fp6,"variables=\"x(m)\"\"y(m)\"\"visc-x\"\"visc-y\"\n");
# print(fp6, "zone T=\"n=%d\"\n",n);
# print(fp6, "I= %d J= %d\n",imax, jmax);
# print(fp6, "DATAPACKING=POINT\n");

# Set derived input quantities
set_derived_inputs()

# Set up headers for output files
output_file_headers()

# Set Initial Profile for u vector
ninit, rtime, resinit = initial(ninit, rtime, resinit)

# Set Boundary Conditions for u
set_boundary_conditions()

# Write out inital conditions to solution file
write_output(ninit, resinit, rtime)

# Initialize Artificial Viscosity arrays to zero (note: artviscx(i,j) and artviscy(i,j)
# np.place(artviscx, artviscx == -99.9, 0)
# np.place(artviscy, artviscy == -99.9, 0)

# Evaluate Source Terms Once at Beginning
# (only interior points; will be zero for standard cavity)
compute_source_terms()

# ========== Main Loop ==========
isConverged = 0

for n in np.arange(ninit, nmax, 1):
    # Calculate time step
    dtmin = compute_time_step(dtmin)

    # Save u values at time level n (u and uold are 2D arrays)
    uold = copy.deepcopy(u)

    if isgs == 1:  # Symmetric Gauss Seidel

        # Artificial Viscosity
        Compute_Artificial_Viscosity()

        # Symmetric Gauss-Siedel: Forward Sweep
        SGS_forward_sweep()

        # Set Boundary Conditions for u
        set_boundary_conditions()

        # Artificial Viscosity
        Compute_Artificial_Viscosity()

        # Symmetric Gauss-Siedel: Backward Sweep
        SGS_backward_sweep()

        # Set Boundary Conditions for u
        set_boundary_conditions()
    elif isgs == 0:  # Point Jacobi

        # Artificial Viscosity
        Compute_Artificial_Viscosity()

        # Point Jacobi: Forward Sweep
        point_Jacobi()

        # Set Boundary Conditions for u
        set_boundary_conditions()
    else:
        print("ERROR: isgs must equal 0 or 1!\n")

    # Pressure Rescaling (based on center point)
    pressure_rescaling()

    # Update the time
    rtime = rtime + dtmin

    # Check iterative convergence using L2 norms of iterative residuals
    res, resinit, conv = check_iterative_convergence(
        n, res, resinit, ninit, rtime, dtmin)

    if conv < toler:
        print(str(fp1)+" "+str(n)+" "+str(rtime)+" " +
              str(res[0])+" "+str(res[1])+" "+str(res[2]))
        isConverged = 1
        break

    # Output solution and restart file every 'iterout' steps
    if n % iterout == 0:
        write_output(n, resinit, rtime)

    # ========== End Main Loop ==========

if isConverged == 0:
    print("Solution failed to converge in "+str(nmax)+" iterations!!!")

if isConverged == 1:
    print("Solution converged in "+str(n)+" iterations!!!")

# Calculate and Write Out Discretization Error Norms (will do this for MMS only)
Discretization_Error_Norms(rL1norm, rL2norm, rLinfnorm)

# Output solution and restart file
write_output(n, resinit, rtime)

# Close open files
fp1.close
fp2.close
#   fclose(fp6); % Uncomment for debug output

PrsMatrix = u[:, :, 0]  # output arrays
uvelMatrix = u[:, :, 1]
vvelMatrix = u[:, :, 2]
