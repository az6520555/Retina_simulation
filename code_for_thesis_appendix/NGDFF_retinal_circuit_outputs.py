import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.io import loadmat
import xlsxwriter 
import import_ipynb
import random
import NGDfunc # the function file should be put in the same directory as this main file
from NGDfunc import MI,xcorr_quick,NGD,poisson,EqualState,OU


# ## Load data / find cross-correlation of ganglion cell output

# generate LPOU stimuli in different contrast
dt=0.01
tau_ou=0.5
fc=1
T_ou=300
nsteps_ou=int(T_ou/dt)
mid_pos=500
randseed = np.random.normal(0, 50, nsteps_ou)
time_ou,x_ou=NGDfunc.OU(T_ou,tau_ou,'no',dt,randseed)
time_ou,x_lpou=NGDfunc.OU(T_ou,tau_ou,fc,dt,randseed)

amp=[0.5,1,2,3]
mean=10
x_set=[]
f_ou,ax_ou=plt.subplots(figsize=(15,5))
for i in range(len(amp)):
    x_set.append(x_lpou/np.std(x_lpou)*amp[i]+mean)
    ax_ou.plot(x_set[i])


# ## NGD model output

# generate outputs of NGD model
alpha,beta,K,g=6,1.6,22,10 # parameters in NGD model
y_set=[NGDfunc.NGD(x_set[j],dt,alpha,beta,K,g,1)[0] for j in range(np.size(x_set,0))] 
z_set=[NGDfunc.NGD(x_set[j],dt,alpha,beta,K,g,1)[1] for j in range(np.size(x_set,0))] 


# f_yz=plt.figure(figsize=(15,10))
# ax_y=f_yz.add_subplot(211)
# for i in range(4):
#     ax_y.plot(y_set[i])
# ax_y.set_ylim(-1,4)
# ax_y.set_xlim()
# ax_y.set_title('y(t)')
# ax_z=f_yz.add_subplot(212)
# for i in range(4):
#     ax_z.plot(z_set[i])
# ax_z.set_title('z(t)')


# ## summing the horizontal feedforward to NGD model output 

# ### MI and cross correlation of B

# generate output u(t) after summing y and z
psi_set=np.array([0.03,0.05,0.1,0.14,0.17,0.2,0.23]) # consider different weightings between y and z
print(psi_set)
iamp=3
x=x_set[iamp]
y=y_set[iamp]
z=z_set[iamp]
B=[(1-psi_set[k])*y-psi_set[k]*z for k in range(len(psi_set))] # u(t) 
MIxyz=[[] for i in range(len(B))]
corrxyz=[[] for i in range(len(B))]
for k in range(len(B)):
    tsxyz,MIxyz[k]=NGDfunc.MI(NGDfunc.EqualState(x,8),NGDfunc.EqualState(B[k],8),dt,[-2,2])
    tc,corrxyz[k]=NGDfunc.xcorr_quick(x,B[k],[5,5],0.01)
# color2=['k','r']
fuout,axuout=plt.subplots()
for i in range(len(B)):
    axuout.plot(tsxyz,MIxyz[i],label=r'$\psi$='+str(round(psi_set[i],3)))
axuout.legend()
axuout.set_xlim(-1,1)
axuout.set_ylabel(r'$MI[x(t),u(t-\delta t)]$ (bits/s)')
axuout.set_xlabel('$\delta t$ (s)')
axuout.axvline(x=0,c='k',linewidth=0.5,linestyle='dashed')
axuout.set_title('MI of x(t) and u(t)')


# fB,axB=plt.subplots(figsize=(15,5))
# for i in range(len(B)):
#     axB.plot(B[i],label=r'$\psi$='+str(round(psi_set[i],3)))
# axB.set_ylim(-5,5)


# NGDFF retinal circuit with different weightings psi
f_onoff_psi=plt.figure(figsize=(15,5))
axon=f_onoff_psi.add_subplot(121)
axoff=f_onoff_psi.add_subplot(122)
for i in range(len(psi_set)):
    spike_psi=poisson(B[i],0,0,dt)
    tl,MI_psi=MI(EqualState(x,8),spike_psi,dt,[-1,1])
    axon.plot(tl,MI_psi,label=r'$\psi$='+str(round(psi_set[i],3)))
axon.legend()
axon.set_title('NGDFF retinal circuit with k=0 (on cell)')
axon.set_xlim(-1,1)
axon.set_ylabel('MI (bit/s)')
axon.set_xlabel(r'$\delta t$ (s)')

for i in range(len(psi_set)):
    spike_psi=poisson(B[i],0,1,dt)
    tl,MI_psi=MI(EqualState(x,8),spike_psi,dt,[-1,1])
    axoff.plot(tl,MI_psi,label=r'$\psi$='+str(round(psi_set[i],3)))
axoff.legend()
axoff.set_title('NGDFF retinal circuit with k=1 (off cell)')
axoff.set_xlim(-1,1)
axoff.set_ylabel('MI (bit/s)')
axoff.set_xlabel(r'$\delta t$ (s)')

# generate u(t) with different stimuli
B_set=[0 for k in range(len(x_set))] # u(t)
for i in range(len(x_set)):
    B_set[i]=y_set[i]*0.87-z_set[i]*0.13

# NGDFF retinal circuit with different stimulus contrasts
spike_on=[0 for i in range(len(x_set))]
spike_off=[0 for i in range(len(x_set))]
th=0
f_NGDFF_cst=plt.figure(figsize=(15,5))
ax2=f_NGDFF_cst.add_subplot(121)
ax3=f_NGDFF_cst.add_subplot(122)
for j in range(len(x_set)):
    spike_on=poisson(B_set[j],th,0,0.01)
    spike_off=poisson(B_set[j],th,1,0.01)
    tl,MI_spike_on=MI(EqualState(x_set[j],8),spike_on,dt,[-2,2])
    tl,MI_spike_off=MI(EqualState(x_set[j],8),spike_off,dt,[-2,2])
    ax2.plot(tl,MI_spike_on,label='C='+str(amp[j]))
    ax3.plot(tl,MI_spike_off,label='C='+str(amp[j]))
ax2.set_xlim(-1,1)
ax3.set_xlim(-1,1)
ax2.set_title('NGDFF retinal circuit with k=0 (on cell)')  
ax3.set_title('NGDFF retinal circuit with k=1 (off cell)')
ax2.legend()
ax3.legend()
ax2.set_ylabel('MI (bit/s)')
ax3.set_ylabel('MI (bit/s)')
ax2.set_xlabel(r'$\delta t$ (s)')
ax3.set_xlabel(r'$\delta t$ (s)')

plt.show()




