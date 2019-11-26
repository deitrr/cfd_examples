import numpy as np
import matplotlib.pyplot as plt
# 1 d advection diffusion by ftcs scheme
# advects and diffuses a sine wave

# user-set parameters
N = 101       # number of grid points
nstep = 1000   # number of time steps
L = 2.0      # size of domain
dt = 0.004  # time step size  (dt < 0.5*dx**2/D and dt < 2*D/U**2)
U = 1       # flow velocity
D = 0.05     # diffusion coeff
k = 1        # wave number
amp = 0.5    # amplitude of wave

# grid
dx = L/(N-1)
x = dx*np.arange(N)

# set up f
time = 0     # initial time
f = amp*np.sin(2*np.pi*k*x)  # initial condition
f[-1] = f[0]

#analytic solution
def soln(x,t):
    return amp*np.exp(-D*(2*np.pi*k)**2*t)*np.sin(2*np.pi*k*(x-U*t))

#set up figure axis
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(x,f,'k-',label='numerical')
exact, = ax.plot(x,soln(x,time),c='0.5',label='exact')
ax.legend(loc='lower left')
ax.set_ylabel('f')
ax.set_xlabel('x')
ax.set_xlim(0,L)
ax.set_ylim(-2*amp,1.5*amp)
txt = ax.text(0.1,1.1*amp,'time = %f'%time)

# integration (time stepping loop)
for m in np.arange(nstep):
    plt.pause(0.1*dt)  #slow it down a little
    ftmp = f.copy() #store tmp copy of f from prev time step
    for j in np.arange(1,N-1):   #spatial loop
        f[j] = ftmp[j]-((0.5*U/dx)*(ftmp[j+1]-ftmp[j-1])-\
                D/dx**2*(ftmp[j+1]-2*ftmp[j]+ftmp[j-1]))*dt
    # periodic boundary condition
    f[N-1] = ftmp[N-1]-((0.5*U/dx)*(ftmp[1]-ftmp[N-2])-\
                D/dx**2*(ftmp[1]-2*ftmp[N-1]+ftmp[N-2]))*dt
    f[0] = f[N-1]
    time = time + dt
    # plot new values and exact solution
    line.set_ydata(f)
    exact.set_ydata(soln(x,time))
    txt.set_text('time = %f'%time)
    fig.canvas.draw()

plt.show()
