import numpy as np
import matplotlib.pyplot as plt
# 2 d advection diffusion by ftcs scheme
# advects and diffusion a constant f from the left boundary

# parameters
Nx = 32      #number of points in x
Ny = 32      #number of points in y
nstep = 140   #number of time steps
D = 0.025    #diffusion coefficient
Lx = 2.0     #size of domain in x
Ly = 2.0     #size of domain in y
dt = 0.02    #time step
u = 1.0      #x-velocity
v = 0.0      #y-velocity

# grid
dx = Lx/(Nx-1)
dy = Ly/(Ny-1)
x, y = np.meshgrid(np.linspace(0,Lx,Nx),np.linspace(0,Ly,Ny))

# initial conditions
time = 0
f = np.zeros((Nx,Ny))
f[0,np.int(np.ceil(Ny/3)):np.int(np.floor(2*Ny/3))]=1.0

#set up figure axis
fig = plt.figure()
ax = fig.add_subplot(111)
im = plt.imshow(f.T,origin='lower',extent=[0,Lx,0,Ly],cmap='cividis')
txt = ax.set_title('time = %f'%time)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im)

for m in np.arange(nstep):
    plt.pause(0.1*dt)
    ftmp = f.copy() #temporary copy of f
    for i in np.arange(1,Nx-1):
        for j in np.arange(1,Ny-1):
            f[i,j] = ftmp[i,j]-((0.5*u/dx)*(ftmp[i+1,j]-ftmp[i-1,j])\
                    +(0.5*v/dy)*(ftmp[i,j+1]-ftmp[i,j-1])\
                    -D*( (ftmp[i+1,j]-2*ftmp[i,j]+ftmp[i-1,j])/dx**2\
                    +(ftmp[i,j+1]-2*ftmp[i,j]+ftmp[i,j-1])/dy**2) )*dt
    f[:,0] = f[:,1]    #zero gradient at lower boundary
    f[-1,:] = f[-2,:]  #zero gradient at right boundary
    f[:,-1] = f[:,-2]  #zero gradient at upper boundary
    time = time+dt
    im.set_data(f.T)
    txt.set_text('time = %f'%time)
    fig.canvas.draw()

plt.show()
