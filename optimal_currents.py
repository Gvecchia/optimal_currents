import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as plt
from matplotlib import cm

# Defnition of the grids

Ls = 0.5; # Square edge length, centered in 0
M = 400;

points = np.linspace(-Ls,Ls, M ,endpoint = False) # M because linspace requires the number of points and
                                                  # ebdpoint = False because we want to use fft
h = points[1] - points[0]
edges = points + 0.5*h  # We shift to the right to take the midpoints (not consider the last point)
points = np.append(points,points[-1] + h) # Now points has M+1 points

xx, yy = np.meshgrid(points,edges) # Recall: cartesian coordinates
meshx = np.stack((xx,yy)) # Mesh for the x component of V

xx, yy = np.meshgrid(edges,points)
meshy = np.stack((xx,yy)) # Mesh for the y component of V

xx, yy = np.meshgrid(edges,edges)
meshc = np.stack((xx,yy)) # Mesh with central points

# Derivative operators fro fft
wavenumbers_1d = ft.fftfreq(M, d = h)*2*np.pi
kxx, kyy = np.meshgrid(wavenumbers_1d,wavenumbers_1d)
wavenumbers = np.stack((kxx,kyy))

derivative_op = 1j*wavenumbers
lapl_op = (derivative_op[0]**2) + (derivative_op[1]**2)
invlapl_op = np.copy(lapl_op) 
invlapl_op[0,0]= -(np.pi/M)**2
invlapl_op = 1/invlapl_op

# Staggered Operators
def midpoint(V):
    # Staggered V. Return midpoint Vector
    Vx = 0.5*(V[0][:,:-1] + V[0][:,1:])
    Vy = 0.5*(V[1][:-1,:] + V[1][1:,:])
    return np.array([Vx, Vy])
    
def staggered(U):
    # from midpoint values to staggered grid, considering vector parallel to the boundary
    Dim = U[0].shape[0]
    Ux = np.zeros((Dim,Dim + 1))
    Uy = np.zeros((Dim + 1,Dim))
    Ux[:,1:-1] = (U[0][:,1:] + U[0][:,:-1])/2
    Uy[1:-1,:] = (U[1][1:,:] + U[1][:-1,:])/2
    return [Ux, Uy]

def div(V,h):
    Vx = V[0]
    Vy = V[1]
    # Staggered V. Return midpoint
    return (Vx[:,1:] - Vx[:,:-1])/h + (Vy[1:,:] - Vy[:-1,:])/h

def diff_xx(Vz,h):
    # Vz can be both Vx or Vy
    # Return midpoint values. Boundary always zero.
    dxxVz = np.zeros_like(Vz)
    dxxVz[:,1:-1] = (Vz[:,2:] + Vz[:,:-2] - 2*Vz[:,1:-1])/(h**2)
    if Vz.shape[0] < Vz.shape[1]: # Vx case
        return  0.5*(dxxVz[:,:-1] + dxxVz[:,1:]) 
    else:
        return 0.5*(dxxVz[:-1,:] + dxxVz[1:,:]) 
    
def diff_yy(Vz,h):
    # Vz can be both Vx or Vy
    # Return midpoint values. Boundary always zero.
    dyyVz = np.zeros_like(Vz)
    dyyVz[1:-1,:] = (Vz[2:,:] + Vz[:-2,:] - 2*Vz[1:-1,:])/(h**2)
    if Vz.shape[0] > Vz.shape[1]: # Vy case
        return  0.5*(dyyVz[:-1,:] + dyyVz[1:,:])
    else:
        return 0.5*(dyyVz[:,:-1] + dyyVz[:,1:])
    
# Energy & Energy gradient

def TE(V,α,h,ϵs,ϵ): # Target Energy function
                       # ϵs is the regularization parameter for the norm. 
                       # Trying to be consistent with notation (almost)
    
    # Parameters
    γ1 = 3*(α - 1)/(α + 1)
    γ2 = 3
    γ = (4*α - 2)/(α + 1)
    
    mV = midpoint(V)
    
    # Norm term
    IV = (h**2)*np.sum( ( mV[0]**2 + mV[1]**2 + ϵs)**(γ/2) )
    
    # Gradient term
    DV = np.sum( (V[0][:,1:] - V[0][:,:-1])**2) + np.sum( (V[0][1:,:] - V[0][:-1,:])**2 ) + np.sum( (V[1][:,1:] - V[1][:,:-1])**2 ) + np.sum( (V[1][1:,:] - V[1][:-1,:])**2 )
    
    return (ϵ**γ1)*IV + (ϵ**γ2)*DV 

def GTE(V,h,α,ϵ,ϵs): # Gradient of the Target Energy function
    # Parameters 
    γ1 = 3*(α - 1)/(α + 1)
    γ2 = 3
    γ = (4*α - 2)/(α + 1)
    
    # V staggered vector. Return midpoint.
    Vx = V[0]
    Vy = V[1]
    GDx = -2*diff_xx(Vx,h) - 2*diff_yy(Vx,h)
    GDy = -2*diff_xx(Vy,h) - 2*diff_yy(Vy,h)
    
    GD = np.stack((GDx,GDy))
    
    mVx, mVy = midpoint(V)
    N2 = mVx**2 + mVy**2 + ϵs
    GIx = γ*mVx*(N2**(γ/2-1))
    GIy = γ*mVy*(N2**(γ/2-1))
    
    GI = np.stack((GIx,GIy))
    
    return (ϵ**(γ1))*GI + (ϵ**γ2)*GD

# Projection Operators

def ProjDiv(V0, f, derivative_op, invlapl_op):
    midV0 = midpoint(V0)
    # Fourier of f - v0
    fftg = ft.fft2(f) - (derivative_op[0]*ft.fft2(midV0[0]) + derivative_op[1]*ft.fft2(midV0[1]))
    # Mult. by inv lapl
    fftphi = invlapl_op*fftg
    # Fix [0,0] entry. It means that we are setting the mean to 0 (it is the constant 'vibration')
    fftphi[0,0] = 0
    # Now I compute the gradient
    fftGphi0 = derivative_op[0]*fftphi 
    fftGphi1 = derivative_op[1]*fftphi
    # Inverse transform and resize
    Gphi = np.zeros_like(midV0)
    Gphi[0,:,:] = np.real(ft.ifft2(fftGphi0))
    Gphi[1,:,:] = np.real(ft.ifft2(fftGphi1))
    Gphi_stag = staggered(Gphi)
    return [V0[0] + Gphi_stag[0], V0[1] + Gphi_stag[1]]

def StableProj(V0,f,derivative_op, invlapl_op, tol, maxiter):
    count = 0
    tol = tol**2 # because we take the square of the L2 norm
    error = tol + 1
    
    while (count < maxiter) and (error > tol):
        V1 = ProjDiv(V0, f, derivative_op, invlapl_op)
        error = (np.sum( (V1[0] - V0[0])**2 ) + np.sum( (V1[1] - V0[1])**2 )) / (np.sum( V1[0]**2 + V0[0]**2 ) + np.sum( V1[1]**2 + V0[1]**2 ))
        V0 = V1
        count += 1
        if count == maxiter:
            print('Maxiter stable projection reached')
    
    return V1  

# MAIN

# Delta function
def approximate_delta_fun(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

def approximate_delta(center,sigma,mesh):
    xc = center[0]
    yc = center[1]
    xx = mesh[0]
    yy = mesh[1]
    return approximate_delta_fun(xx - xc, yy - yc, sigma)

# Functions and parameter initialization
α = 0.65
ϵ = 1e-4 # h**2
ϵs = 1e-4 #h**2


PS =  approximate_delta([0,0],0.005,meshc)
NS = 0.5*approximate_delta([0.1,-0.3],0.005,meshc) + 0.5*approximate_delta([-0.1,-0.3],0.005,meshc)
S = PS - NS;
V0x = np.zeros(meshx[0].shape)
V0y = np.zeros(meshy[0].shape)
V0 = [V0x,V0y]

V1 = ProjDiv(V0, S, derivative_op, invlapl_op) # Initialisation
V0 = V1

tol_fun = 1e-10
tol_prj = 1e-4 

maxiter_fun = 2 # 100000
maxiter_prj = 2 #10000

τ = h # Gradient step

# Computation of the optimal function

count = 0
error = tol_fun + 1
energy = np.array([])

while (count < maxiter_fun) and (error > tol_fun):
    energy0 = TE(V0,α,h,ϵs,ϵ)
    GE = GTE(V0,h,α,ϵ,ϵs)
    GE = staggered(GE)
    
    V1 = StableProj([V0[0] - τ*GE[0] , V0[1] - τ*GE[1]], S, derivative_op, invlapl_op, tol_prj, maxiter_prj)
    energy1 = TE(V1,α,h,ϵs,ϵ)
    
    count_prj = 0
    while (count_prj < maxiter_prj) and (energy0 < energy1):
        τ = τ/2
        print(τ)
        V1 = StableProj([V0[0] - τ*GE[0] , V0[1] - τ*GE[1]], S, derivative_op, invlapl_op, tol_prj, maxiter_prj)
        energy1 = TE(V1,α,h,ϵs,ϵ)
        count_prj += 1
        if count_prj == maxiter_prj:
            print('Maxiter Prj iter reached at: ', count)
    
    energy = np.append(energy,energy1)
    error = (np.sum( (V1[0] - V0[0])**2 ) + np.sum( (V1[1] - V0[1])**2 )) / (np.sum( V1[0]**2 + V0[0]**2 ) + np.sum( V1[1]**2 + V0[1]**2 ))
    V0 = V1
    count += 1
    if count == maxiter_fun:
        print('Maxiter function reached with error: ', error)

V = midpoint(V1)
fig = plt.figure()
plt.contourf(xx,yy, (V[0]**2 + V[1]**2)**0.5 )
plt.colorbar()
plt.show()