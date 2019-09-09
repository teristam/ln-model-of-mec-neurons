"""
Create EEG file and extract the position file for faster processing
"""

#%%
import OpenEphys
import setting
import numpy as np
import scipy.signal as signal

#%% Specifiy the input and output
#make provisions such that the script can be run independently or through snakemake

if 'snakemake' not in locals():
    input = [f'test/100_CH{x}.continuous' for x in range(1,setting.numChan+1)]
    input.append(f'test/100_{setting.positionChannel}.continuous')
    output = ['test/processing/eeg.npy','test/processing/position.npy']
else:
    input = snakemake.input
    output = snakemake.output


#%% Load and downsample to create data

data=[]
for fn in input[:-1]:
    d = OpenEphys.loadContinuousFast(fn)['data']
    data.append(signal.decimate(d,setting.Fs//setting.eeg_Fs,ftype='fir'))


#%% Convert to numpy array and save
rawSignal = np.vstack(data).T
np.save(output[0],rawSignal)


#%% Load and save position data
d = OpenEphys.loadContinuousFast(input[-1])['data']
np.save(output[1],d)

#%%
# d = OpenEphys.loadContinuousFast('test/100_CH2.continuous')['data']
# h = signal.welch(d,fs=30000)
# plt.plot(h[0][:10],h[1][:10])

#%%
