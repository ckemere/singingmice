import os
import numpy as np
import ghostipy as gsp
import time
from cProfile import Profile
from pstats import SortKey, Stats

# This is a python clone of the buzcode bz_LFPfromDat.m - https://github.com/buzsakilab/buzcode/blob/master/io/bz_LFPfromDat.m

# Step 1: find files in current directory to process

raw_filename = "amplifier.dat" # perhaps should look somewhere else?

n_channels = 128 # should infer from info.rhd

## TODO:bz_LFPfromDat also creates a session file

# Step 2: figure out dimenions of data file
raw_file_stats = os.stat(raw_filename)
n_samples = int(raw_file_stats.st_size / 2 / n_channels) # each sample is 2 bytes

# Step 3:

fs = 30000
low_pass_fs = 450

lfp_fs = 1250

filter_length = 1025
filter = np.sinc(np.arange(-int(filter_length/2), int(filter_length/2)+1)*(low_pass_fs/(fs/2)))

ds = int(fs/lfp_fs) # should warn if not an int!

polyphase_filters = []
for idx in range(ds):
   if (idx == 0):
      polyphase_filters.append(filter[::ds])
   else:
      polyphase_filters.append(filter[(ds-idx)::ds])

# input = np.memmap(raw_filename, dtype='uint16', mode = 'r', shape=(n_samples, n_channels), order='C')

# We're going to do something cool. Assuming n_channels is 127, the input data is laid out as
#      [ch0,sample0] [ch1,sample0] ... [ch127,sample0] [ch0,sample1], [ch1,sample1], etc
# Because of this, we can implement the input decimation part of the polyphase filter by reshaping
# the memmap to be {ds} times wider. So then when we apply the filters, we apply them in sequence to the first
# ds "channels" then the second ds "channels" and so on. In order to have this memmap be smaller than the
# actual filesize, we'll have to truncate it by a fraction of {ds} samples, meaning that the last few samples
# of our output may be corrupted a bit...

# with Profile() as profile:
indata = np.memmap(raw_filename, dtype='uint16', mode = 'r', shape=(int(n_samples/24), n_channels*24), order='C')

print(indata.shape)
t0= time.time()

filter_inds = np.arange(n_channels, dtype=int)
output = gsp.filter_data_fir(indata, polyphase_filters[0], axis=0, input_dim_restrictions=[None,filter_inds.tolist()])
t1= time.time()
print(t1 - t0, 's')


for idx in range(ds-1):
   filter_inds = filter_inds + 128
   temp = gsp.filter_data_fir(indata, polyphase_filters[idx+1], axis=0, input_dim_restrictions=[None,filter_inds.tolist()])
   t1= time.time()
   print(t1 - t0, 's')
   output[1:,:] += temp[:(output.shape[0]-1),:]
   t1= time.time()
   print(t1 - t0, 's')

print(output.shape)

indata = np.memmap(raw_filename, dtype='uint16', mode = 'r', shape=(n_samples, n_channels), order='C')


t0= time.time()

output = gsp.filter_data_fir(indata, filter, axis=0)
print(output.shape)


t1= time.time()
print(t1 - t0, 's')

   # Stats(profile).strip_dirs().sort_stats(SortKey.CALLS).print_stats()

