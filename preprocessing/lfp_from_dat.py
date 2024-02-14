import os
import numpy as np
import ghostipy as gsp
import time
from cProfile import Profile
from pstats import SortKey, Stats

# This is a python clone of the buzcode bz_LFPfromDat.m - https://github.com/buzsakilab/buzcode/blob/master/io/bz_LFPfromDat.m

# Step 1: find files in current directory to process

raw_filename = "amplifier.dat" # perhaps should look somewhere else? TODO: Specify path

n_channels = 128 # should infer from info.rhd

## TODO:bz_LFPfromDat also creates a session file
_, session_dir  = os.path.split(os.getcwd()) # Assumes we are running from the data directory. TODO: FRAGILE

# Step 2: figure out dimenions of data file
raw_file_stats = os.stat(raw_filename)
n_samples = int(raw_file_stats.st_size / 2 / n_channels) # each sample is 2 bytes

# Step 3: Filter and downsample

fs = 30000
low_pass_fs = 450

lfp_fs = 1250

filter_length = 1025
filter = np.sinc(np.arange(-int(filter_length/2), int(filter_length/2)+1)*(low_pass_fs/(fs/2)))

ds = int(fs/lfp_fs) # should warn if not an int!

indata = np.memmap(raw_filename, dtype='uint16', mode = 'r', shape=(n_samples, n_channels), order='C')

output_shape, output_dtype = gsp.filter_data_fir(indata, filter, axis=0, describe_dims=True, ds=ds)

output_filename = session_dir + '_' + 'lfp.dat'

# The original plan was to use a polyphase filter, but for shorter files its just safer/easier
# to do this. On my M2 MBAir, we process about 3.25 GB/minute (which corresponds to about 7 minutes of recording).

outdata = np.memmap(output_filename, dtype='uint16', mode = 'w+', shape=output_shape, order='C')
print("Output to: ", output_filename)


t0= time.time()
output = gsp.filter_data_fir(indata, filter, axis=0, ds=ds, outarray=outdata)
t1= time.time()
print(t1 - t0, 's')

print('Flushing data to disk.')
outdata.flush()
t1= time.time()
print(t1 - t0, 's')

   # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

