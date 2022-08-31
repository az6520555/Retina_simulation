# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython


#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#get_ipython().run_line_magic('matplotlib', 'inline')


#%%
mat_contents=sio.loadmat('20190408_HMM_G5_sort_unit1.mat')
Spikes = mat_contents['Spikes']
a_data = mat_contents['a_data']
TimeStamps = mat_contents['TimeStamps']
print(Spikes.dtype)
print(Spikes[0,0].dtype)
print(TimeStamps.shape)


#%%



#%%
print(a_data)
print(a_data.shape)
print(a_data[2,0:])


#%%
a_data3=a_data[2,0:]
rate=20000
t=np.arange(1,len(a_data3)+1)/rate
print(t)
print(t.shape)
plt.plot(t,a_data3)
plt.show


#%%
tcut=t[(t>TimeStamps[0,0]) & (t<TimeStamps[0,1])]-TimeStamps[0,0]
Xsti=a_data3[(t>TimeStamps[0,0]) & (t<TimeStamps[0,1])]
print(tcut)
plt.plot(tcut,Xsti)
plt.show


#%%
# normalize Xstimulation 
Xsti=Xsti-np.mean(Xsti)
Xsti=Xsti/np.std(Xsti)
plt.plot(tcut,Xsti)
#plt.xlim(100,100.001)
#plt.ylim(-2,0)
plt.show


#%%
# speed of Xstimulation
Vsti=np.diff(Xsti)
Vsti=Vsti/np.std(Vsti)
plt.plot(tcut[0:-1],Vsti)
plt.show()


#%%
print(tcut.shape)


#%%
# spikes binning
"""bininterval=0.05
bintime=np.arange(0,tcut[-1],bininterval)
print(Spikes.shape)
SpikesBinningAll=[]
for i in range(60):
    SpikesSingleChannel=[]
    SpikesBinning=[]
    SpikesSingleChannel=np.squeeze(Spikes[0,i])
    SpikesBinning=np.histogram(SpikesSingleChannel,bins=bintime)
    SpikesBinningAll.append(SpikesBinning)"""


#%%
#plt.plot(SpikesBinningAll[1][0])
#plt.show()


#%%
# find spike trigger feature
delaytime=0.5
delayind=delaytime*rate
Spikes2=[]
indAll=[]
for i in range(60):
    SpikesSingleChannel=[]
    SpikesSingleChannel=np.squeeze(Spikes[0,i])
    indSpikes=np.round(SpikesSingleChannel*rate,0)
    indStiFeature=indSpikes-delayind
    indStiFeature=indStiFeature[(indStiFeature>0) & (indStiFeature<=len(tcut))]
    indStiFeature=indStiFeature.astype(int)
    indAll.append(indStiFeature)
#plt.plot(Xsti[indAll[20]],Vsti[indAll[20]])
for i in range(60):
    plt.hist2d(Xsti[indAll[i]], Vsti[indAll[i]], bins=(100,100), cmap=plt.cm.Greys)
    plt.grid()
    plt.colorbar()
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.show()

