import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
# 1-D shock tube problem
# numerically solves with a first order Lax scheme
# uses an adaptive time step given by cfl condition
# compares to analytical solution from Hawley+ 1984

# parameters
Nx = 1001
x0 = np.int((Nx-1)/2)    # location of barrier
dx = 1e-3                # grid spacing
tfinal = 0.25            #final time
nplt = 100               #number of plot frames
gamma = 5/3      # adiabatic exponent
mu2 = (gamma-1)/(gamma+1)
rhoL = 1         # density left of barrier
rhoR = 0.125     # density right of barrier
PL = 1           # pressure left of barrier
PR = 0.1         # pressure right of barrier

# initial conditions and bean-counting
time = 0                 #store current time
next = time + tfinal/nplt
stop = 0
x = np.linspace(0,(Nx-1)*dx,Nx)  #grid
rho = np.zeros(Nx)       #density
rho[:x0+1] = rhoL
rho[x0+1:] = rhoR
p = np.zeros(Nx)         #pressure
p[:x0+1] = PL
p[x0+1:] = PR
v = np.zeros(Nx)         #velocity
rhov = rho*v             #momentum
energy = p/(gamma-1) + 0.5*rho*v**2      #energy
cs = np.sqrt(gamma*p/rho)                #sound speed
csL = np.sqrt(gamma*PL/rhoL)  #initial sound speed to left of barrier
dt = 0.5*np.min(dx/np.sqrt(cs**2+v**2))  #cfl condition for dt

# riemann solution from Hawley+ 1984
def post_shock_p(pps):
    # equation for finding post shock pressure
    # need to solve post_shock_p(pps) = 0
    return (1-mu2**2)**0.5/mu2*(PL/rhoL)**(1/(2*gamma))*\
            (PL**((gamma-1)/(2*gamma))-pps**((gamma-1)/(2*gamma)))\
            -(pps-PR)*((1-mu2)/(rhoR*(pps+mu2*PR)))**0.5

 # some quantities are static so just calculate once
pps = opt.fsolve(post_shock_p,PR)  # find post shock pressure numerically
                                   # i am cheating and using scipy's solver
vps = (pps-PR)*((1-mu2)/(rhoR*(pps+mu2*PR)))**0.5  # post shock velocity
rhops = (mu2*PR+pps)/(mu2*pps+PR)*rhoR             # post shock density
rhomid = rhoL*(pps/PL)**(1/gamma)                  # density in mid region
csmid = np.sqrt(gamma*pps/rhomid)
vsh = vps/(1-rhoR/rhops)                           # shock front velocity
def riemann_soln(time):
    #solution as a function of time
    vrie = np.zeros_like(x)
    prie = np.zeros_like(x)
    rho_rie = np.zeros_like(x)
    energy_rie = np.zeros_like(x)
    #calculate locations of regions
    xrare = x0*dx - csL*time               # head of rarefaction wave
    xmid = (vps/(1-mu2)-csL)*time + x0*dx  # left boundary of mid region
    xcd = x0*dx + vps*time                 # contact discontinuity
    xsh = x0*dx + vsh*time                 # shock front
    #stitch solution together
    ileft = (x<xrare)                      # mask for left region
    irarew = np.logical_and(x>=xrare,x<xmid) #mask for rarefaction wave region
    imid = np.logical_and(x>=xmid,x<xcd)   # mask for middle region
    ipostsh = np.logical_and(x>=xcd,x<xsh) # mask for post shock region
    iright = (x>=xsh)                      # mask for pre shock region
    #velocity
    vrie[irarew] =(1-mu2)*((x[irarew]-x0*dx)/time+csL)
    vrie[imid] = vps
    vrie[ipostsh] = vps
    #density
    rho_rie[ileft] = rhoL
    rho_rie[irarew] = rhomid*(1-(gamma-1)/2*(vrie[irarew]-vps)/csmid)**(2/(gamma-1))
    rho_rie[imid] = rhomid
    rho_rie[ipostsh] = rhops
    rho_rie[iright] = rhoR
    #pressure
    prie[ileft] = PL
    prie[irarew] = pps*(1-(gamma-1)/2*(vrie[irarew]-vps)/csmid)**(2*gamma/(gamma-1))
    prie[imid] = pps
    prie[ipostsh] = pps
    prie[iright] = PR
    #energy
    energy_rie = prie/(gamma-1) + 0.5*rho_rie*vrie**2

    return vrie, energy_rie, rho_rie, prie

#set up figure axes
fig = plt.figure(figsize=(12,7))
ax1 = fig.add_subplot(221)
line1, = ax1.plot(x,v)
ax1.set_ylabel('Velocity')
ax1.set_xlabel('x')
ax1.set_ylim(0,1)
txt = ax1.text(0.1,0.9,'time = %f'%time)
line1ex, = ax1.plot(x,v,'k--')

ax2 = fig.add_subplot(222)
line2, = ax2.plot(x,p/(gamma-1)/rho,label='Lax method')
ax2.set_ylabel('Internal Energy (per unit mass)')
ax2.set_xlabel('x')
ax2.set_ylim(0.5,2.0)
line2ex, = ax2.plot(x,p/(gamma-1)/rho,'k--',label='Analytical solution')
plt.legend(loc='lower left')

ax3 = fig.add_subplot(223)
line3, = ax3.plot(x,rho)
ax3.set_ylabel('Density')
ax3.set_xlabel('x')
ax3.set_ylim(0,1.1)
line3ex, = ax3.plot(x,rho,'k--')

ax4 = fig.add_subplot(224)
line4, = ax4.plot(x,p)
ax4.set_ylabel('Pressure')
ax4.set_xlabel('x')
ax4.set_ylim(0,1.1)
line4ex, = ax4.plot(x,p,'k--')

# time loop
# integration variables are rho, rhov, energy
while stop == 0:
    time += dt
    rho_tmp = rho.copy()
    rhov_tmp = rhov.copy()
    energy_tmp = energy.copy()
    for i in np.arange(1,Nx-1):
        #continutiy equation
        rho[i] = 0.5*(rho_tmp[i+1]+rho_tmp[i-1]) \
                -0.5*(rhov_tmp[i+1]-rhov_tmp[i-1])*dt/dx

        #momentum (Euler) equation
        rhov[i] = 0.5*(rhov_tmp[i+1]+rhov_tmp[i-1]) \
                 -0.5*( (rhov_tmp[i+1]*v[i+1]-rhov_tmp[i-1]*v[i-1])\
                      + (p[i+1] - p[i-1]) )*dt/dx

        #energy equation
        energy[i] = 0.5*(energy_tmp[i+1]+energy_tmp[i-1]) \
                    -0.5*( v[i+1]*(energy_tmp[i+1]+p[i+1])-
                           v[i-1]*(energy_tmp[i-1]+p[i-1]) )*dt/dx

    #boundaries
    rhov[0] = rhov[1]
    rhov[-1] = rhov[-2]
    rho[0] = rho[1]
    rho[-1] = rho[-2]
    energy[0] = energy[1]
    energy[-1] = energy[-2]

    #recalculate other quantities
    v = rhov/rho
    p = (energy - 0.5*rhov*v)*(gamma-1)
    cs = np.sqrt(gamma*p/rho)

    #check time step size and adjust if needed
    dtmin = 0.5*np.min(dx/np.sqrt(cs**2+v**2))
    if dt > dtmin:
        dt = dtmin

    #nearing the end... do one more step
    if time+dt > tfinal:
        dt = tfinal - time

    #stop now
    if time >= tfinal:
        stop = 1
        next = time

    # update figure
    if time >= next:
        next = time + tfinal/nplt
        plt.pause(0.001)
        line1.set_ydata(v)
        line2.set_ydata(p/(gamma-1)/rho)
        line3.set_ydata(rho)
        line4.set_ydata(p)
        txt.set_text('time = %f'%time)
        vrie, energy_rie, rho_rie, prie = riemann_soln(time)
        line1ex.set_ydata(vrie)
        line2ex.set_ydata(prie/(gamma-1)/rho_rie)
        line3ex.set_ydata(rho_rie)
        line4ex.set_ydata(prie)
        fig.canvas.draw()

plt.show()
