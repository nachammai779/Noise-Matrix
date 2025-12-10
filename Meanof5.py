import numpy as np
XAVG = np.full((5, 512, 512, 55), np.nan, dtype=np.float16)

list=[]
f=open('/nvme2tb/nachammai/FeatureExtraction/AVGcov441.lst','r')
for l in f.readlines():
    list.append(l.strip().split()[0])
    
fpath = '/nvme2tb/nachammai/FeatureExtraction/AVGpdnet55/'
fpathsfx = '.pdnet55.npy'
for i in range(5):
    fd=[]
    fd=np.load(fpath + list[i] + fpathsfx)
    XAVG[i,:,:,:] = fd
    
output = np.nanmean(XAVG, axis=0)

np.save( '/nvme2tb/nachammai/FeatureExtraction/AVGpdnet55/' + '250AVG.pdnet55.npy', output.astype(np.float16))