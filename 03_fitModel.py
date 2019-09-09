# Fit the GLM model 
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
    input = ['test/processing/binned_data.pkl']
    output = ['test/processing/fitted_models.pkl']
else:
    input = snakemake.input
    output = snakemake.output


#%% Load previous results
data = pickle.load(open(input[0],'rb'))

#%%
