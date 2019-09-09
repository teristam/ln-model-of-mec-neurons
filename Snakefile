import setting 
#don't use config to store your setting, it will cause conflict because snakemake is also using config

# Aggregate data
rule make_eeg:
    input: 
        [f'test/100_CH{x}.continuous' for x in range(1,setting.numChan+1)],
        f'test/100_{setting.positionChannel}.continuous'
    output:
        'test/processing/eeg.npy',
        'test/processing/position.npy'
    script:
        '00_aggregateData.py'

#convert those data to bins
rule bin_data:      
    input:
        'test/processing/eeg.npy',
        'test/processing/position.npy',
        'test/spatial_firing.pkl'
    output:
        'test//processing/binned_data.pkl'
    script:
        '01_binData.py'
