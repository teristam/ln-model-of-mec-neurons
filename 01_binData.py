#%%
import setting
import numpy as np
import pickle
import scipy.signal as signal
import scipy.io as sio
import GLMPostProcessor
import pandas as pd
import tqdm

if 'snakemake' not in locals():
    input = ['test/processing/eeg.npy',
        'test/processing/position.npy',
        'test/spatial_firing.pkl']
    output = ['test/processing/binned_data.pkl']
else:
    input = snakemake.input
    output = snakemake.output

#%% load files
eeg = np.load(input[0])
position = np.load(input[1])
dataframe = pickle.load(open(input[2],'rb'))

#%% Bin location
position_ds = signal.decimate(position, setting.Fs//setting.eeg_Fs,ftype='fir')
position_corrected=GLMPostProcessor.correct_for_restart(position_ds)
position_cm = GLMPostProcessor.calculate_track_location(position_corrected,200)

idx2delete = GLMPostProcessor.findTeleportPoint(position_cm) #delete the teleportation points
position_bin = np.delete(position_cm,idx2delete)

(posgrid,postvec) = GLMPostProcessor.pos_1d_map(position_bin, setting.position_bin, setting.trackLength)

#%% Bin spiketrain
binEdge = np.arange(0,position.shape[0]+1,setting.binSize).astype(int) #make sure it converts the last spike

spiketrain_bin = []

for i in range(len(dataframe)):
    ft=dataframe.iloc[i]['firing_times']
    spiketrain = GLMPostProcessor.make_binary_bin(ft,binEdge)
    spiketrain=np.delete(spiketrain,idx2delete) #delete teleportation points
    spiketrain_bin.append(spiketrain)

spiketrain_bin = np.stack(spiketrain_bin)

#%% Bin EEG phase
#  FIlter and extract phase of eeg
phase = GLMPostProcessor.extract_theta_phase(eeg,np.array([4,12]),setting.eeg_Fs)
phases = []
for i in range(phase.shape[1]):
    phases.append(np.delete(phase[:,i],idx2delete))

phases = np.stack(phases).T

# Bin theta phase into maps
thetagrids =[]

for i in range(phases.shape[1]):
    (thetagrid,thetavect)=GLMPostProcessor.theta_map(phases[:,i].ravel(),18)
    thetagrids.append(thetagrid)

thetagrids = np.stack(thetagrids)


#%% Bin speed

(speedgrid,speedvec,speed) = GLMPostProcessor.speed_map_1d(position_cm,18,setting.Fs/setting.binSize,maxSpeed=50,removeWrap=True)


#%% Save everything
data2save = {
    'speed_bin':speedgrid,
    'speedvec': speedvec,
    'speed': speed,
    'theta_bin': thetagrids,
    'phases': phases,
    'spiketrain_bin': spiketrain_bin,
    'position_bin':posgrid,
    'position_cm': position_cm,
    'posvec': postvec
}

pickle.dump(data2save, open(output[0],'wb'))




#%%
