#%%
import setting
import numpy as np
import pickle
import scipy.signal as signal
import scipy.io as sio
import GLMPostProcessor
import pandas as pd

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

#%% FIlter and extract phase of eeg
phase = GLMPostProcessor.extract_theta_phase(eeg,np.array([4,12]),setting.eeg_Fs)

#%% Bin theta phase into maps
(thetagrid,thetavect)=GLMPostProcessor.theta_map(phase[0,:].ravel(),18)

#%% Bin location
position_corrected=GLMPostProcessor.correct_for_restart(position)
position_cm = GLMPostProcessor.calculate_track_location(position_corrected,200)

binEdge = np.arange(setting.binSize,position_cm.shape[0],setting.binSize).astype(int) #make sure it converts the last spike
position_bin = GLMPostProcessor.average_in_bin(position_cm,binEdge,setting.binSize)
idx2delete = GLMPostProcessor.findTeleportPoint(position_bin) #delete the teleportation points
position_bin = np.delete(position_bin,idx2delete)

#%% Bin spiketrain
spiketrains = []

for i in range(len(dataframe)):
    ft=dataframe.iloc[i]['firing_times']
    spiketrain = GLMPostProcessor.make_binary_bin(ft,binEdge)
    spiketrain=np.delete(spiketrain,idx2delete) #delete teleportation points
    spiketrains.append(spiketrain)


#%%
