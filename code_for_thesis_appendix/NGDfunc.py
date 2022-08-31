import numpy as np
import math
from scipy import signal
from scipy.fftpack import fft, fftfreq
from scipy.io import loadmat
import random


def NGD(x,dt,alpha,beta,K,g,phi): # NGD model
    y=np.zeros(len(x))
    z=np.zeros(len(x))
    for j in range(len(x)-1):
        dy=dt*(-alpha*y[j]+K*(x[j]-phi*z[j]))
        dz=dt*(-beta*z[j]+g*y[j])
        y[j+1]=y[j]+dy
        z[j+1]=z[j]+dz
    return y,z
def EqualState(x, num_state): # assign data into several states
    xs=np.sort(x)
    binlen=int(len(x)/num_state-0.5) #round
    edges = xs[np.arange(num_state)*binlen]
    xstate=np.zeros(len(x))
    for i in range(num_state):
        xstate[x>=edges[i]] = i
    xstate = xstate.astype(int)
    return xstate
def MI(xstate,ystate,dt,window): # mutual information calculation
    negshift=window[0] # second
    posshift=window[1] # second
    shiftdu=dt # second
    shiftlen=(posshift-negshift)/dt+1
    timeshift=np.linspace(negshift,posshift,int(shiftlen))
    bitshift=np.linspace(negshift/dt,posshift/dt,int(shiftlen))
    xedges=np.arange(min(xstate),max(xstate)+0.0001)
    yedges=np.arange(min(ystate),max(ystate)+1+0.0001)
    
    # shifted data
    # shift>0 => y shifted to positive side
    MIvalue=np.zeros(len(bitshift))
    for i in range(len(bitshift)):
        xx=[]
        yy=[]
        shift=int(bitshift[i])
        if shift>0:
            xx=xstate[shift:]
            yy=ystate[:-shift]
        elif shift==0:
            xx=xstate
            yy=ystate
        elif shift<0:
            xx=xstate[:shift]
            yy=ystate[-shift:]

        H, xedges, yedges = np.histogram2d(xx, yy, bins=(xedges, yedges))
        statesum=np.sum(H)
        px_list=np.sum(H,axis=1)/statesum
        py_list=np.sum(H,axis=0)/statesum
        pxy_list=H/statesum

        MIsingle=np.zeros((len(px_list),len(py_list)))
        for ix in range(len(px_list)):
            for iy in range(len(py_list)):
                if pxy_list[ix][iy]==0:
                    MIsingle[ix][iy]=0
                else:
                    MIsingle[ix][iy]=pxy_list[ix][iy]*math.log2(pxy_list[ix][iy]/px_list[ix]/py_list[iy])/dt
        MIvalue[i]=np.sum(MIsingle)
    return timeshift,MIvalue
# generate OU
def OU(Tot,tau,fc,dt,*args): # generate OU stimuli 
    T=np.arange(dt,Tot,dt)
    D = 4
    L=np.zeros(len(T))
    if not args:
        Seed=np.random.normal(0,1,len(T))
    else:
        Seed=args[0]
    for i in range(len(T)-1):
        L[i+1]=L[i]*(1-dt/tau)+math.sqrt(D*dt)*Seed[i]

    # filtering
    if fc!='no':
        b, a = signal.butter(2, 2*fc*dt, btype='low', analog=False)
        Lf = signal.filtfilt(b, a, L)
        x=Lf
    else:
        x=L
    return T,x

def FFT_quick(data,dt): # Fast Fourier transform 
    Ts=dt
    xft=fft(data)
    xftreal = xft.real  
    xftimg = xft.imag
    xft_abs=abs(xft)
    xft_nor=xft_abs/len(xft_abs)
    xft_nor2=xft_nor[range(int(len(xft_nor)/2))]
    freq=fftfreq(len(data), d=dt)
    freq=freq[range(int(len(freq)/2))]
    phase=np.angle(xft)
    phase=phase[range(int(len(phase)/2))]
    
    return freq,xft_nor2,phase
def xcorr_quick(x,y,window,dt): # cross correlation
    lags=np.arange(int(window[0]/dt),int(window[1]/dt)+1e-5)
    lags=lags.astype(int)
    corr=np.zeros(len(lags))
    timelag=lags*dt
    for icorr in range(len(lags)):
        if lags[icorr]<0:
            corr[icorr]=np.sum(x[:lags[icorr]]*y[-lags[icorr]:])#/len(x[:-1+lags[icorr]])
        elif lags[icorr]==0:
            corr[icorr]=np.sum(x*y)#/len(x)
        else:
            corr[icorr]=np.sum(x[lags[icorr]:]*y[:-lags[icorr]])#/len(x[lags[icorr]:])
    return timelag,corr

# stochastic spike generating process 
# ON cell: onoff=0
# OFF cell: onoff=1
def poisson(r_in,thr,onoff,dt): 
    rtemp=(-1)**onoff*(r_in-thr)
    rtemp[rtemp<0]=0
    rtemp=rtemp/(np.sum(rtemp[100:]))*10*len(rtemp[100:]) # set the mean firing rate as 5 hz
    tempspikes=np.zeros(len(rtemp))
    for step in range(len(rtemp)):
        if random.random()<rtemp[step]*dt:
            tempspikes[step]=1
        else:
            tempspikes[step]=0
    return tempspikes


