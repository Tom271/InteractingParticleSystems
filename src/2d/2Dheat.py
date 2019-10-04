import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import scipy.stats as stats

def solve_heat_eqn(D=1, dt=0.01, dx=0.1, dy=0.1, T_end=1, Lx=1, Ly=1, U_0x=0, U_0y=0,  U_Lx=0, U_Ly=0,
            initial_dist=(lambda x,y : x)):


    t = np.linspace(0, T_end, T_end/dt +1)
    x = np.linspace(0, Lx , Lx/dx +1)
    y = np.linspace(0, Ly , Ly/dy +1)

    Nt = len(t)
    Nx = len(x)
    Ny = len(y)
    Fx =  D*dt/dx**2
    Fy = D*dt/dy**2

    U   = np.zeros((Nx+1, Ny+1))      # unknown u at new time level
    U_n = np.zeros((Nx+1, Ny+1))      # u at the previous time level

    Ix = range(0, Nx)
    Iy = range(0, Ny)
    It = range(0, Nt)

    # Make U_0x, U_0y, U_Lx and U_Ly functions if they are float/int
    # Allows bdy = 0 rather than lambda t: 0
    if isinstance(U_0x, (float,int)):
        _U_0x = float(U_0x)  # Make copy of U_0x
        U_0x = lambda t: _U_0x
    if isinstance(U_0y, (float,int)):
        _U_0y = float(U_0y)  # Make copy of U_0y
        U_0y = lambda t: _U_0y
    if isinstance(U_Lx, (float,int)):
        _U_Lx = float(U_Lx)  # Make copy of U_Lx
        U_Lx = lambda t: _U_Lx
    if isinstance(U_Ly, (float,int)):
        _U_Ly = float(U_Ly)  # Make copy of U_Ly
        U_Ly = lambda t: _U_Ly

    for i in Ix:
        for j in Iy:
            U_n[i,j] = initial_dist(x[i], y[j])

    # Data structures for the linear system
    N = (Nx+1)*(Ny+1)  # no of unknowns
    A = np.zeros((N, N))
    b = np.zeros(N)

    # Fill in dense matrix A, mesh line by line (mapping defn)
    m = lambda i, j: j*(Nx+1) + i

    #Fill in bottom bdy points
    j = 0
    for i in Ix:
        p = m(i,j);  A[p, p] = 1
    #Interior
    for j in Iy[1:-1]:
        i = 0;  p = m(i,j);  A[p, p] = 1   # boundary
        for i in Ix[1:-1]:                 # interior points
            p = m(i,j)
            A[p, m(i,j-1)] = - 0.5*Fy
            A[p, m(i-1,j)] = - 0.5*Fx
            A[p, p]        = 1 + (Fx+Fy)
            A[p, m(i+1,j)] = - 0.5*Fx
            A[p, m(i,j+1)] = - 0.5*Fy
        i = Nx;  p = m(i,j);  A[p, p] = 1  # boundary
    # Top bdy
    j = Ny
    for i in Ix:
        p = m(i,j);  A[p, p] = 1

    # Time loop
    #Solve in time, calculates new b vector for solve
    for n in It[0:-1]:
        # Compute b
        j = 0
        for i in Ix:
            p = m(i,j);  b[p] = U_0y(t[n+1])  # boundary
        for j in Iy[1:-1]:
            i = 0;  p = p = m(i,j);  b[p] = U_0x(t[n+1])  # boundary
            for i in Ix[1:-1]:
                p = m(i,j)                                # interior
                b[p] = U_n[i,j] + \
                  0.5*(
                  Fx*(U_n[i+1,j] - 2*U_n[i,j] + U_n[i-1,j]) +\
                  Fy*(U_n[i,j+1] - 2*U_n[i,j] + U_n[i,j-1]))
            i = Nx;  p = m(i,j);  b[p] = U_Lx(t[n+1])     # boundary
        j = Ny
        for i in Ix:
            p = m(i,j);  b[p] = U_Ly(t[n+1])  # boundary

        c = scipy.linalg.solve(A, b)
        # Fill u with vector c (reverse the mapping)
        for i in Ix:
            for j in Iy:
                u[i,j] = c[m(i,j)]
        U_n, U = U, U_n

    return t, U

t,u = solve_heat_eqn(initial_dist=(lambda x,y: np.sin(np.pi*x)*np.sin(np.pi*y)))

X = np.linspace(-Lx, Lx+dx , 2*Lx//dx)
Y = np.linspace(-Ly, Ly+dy , 2*Ly//dy)
X,Y = np.meshgrid(X,Y)
plt.plot_surface(X,Y,u)
