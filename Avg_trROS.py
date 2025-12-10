import numpy as np
X512 = np.full((50, 512, 512, 526), np.nan, dtype=np.float16)

list=[]
f=open('/nvme2tb/nachammai/FeatureExtraction/Fifth_50.lst','r')
for l in f.readlines():
    list.append(l.strip().split()[0])
    
fpath = '/backup/pdnet-feature-analysis/gen-feats/data/deepcov/trRos/'
fpathsfx = '.npy'
for i in range(50):
    fd=[]
    fd=np.load(fpath + list[i] + fpathsfx)
    L=len(fd[:,0,0])
    X512[i,:L,:L,:] = fd
    
output = np.nanmean(X512, axis=0)

np.save( '/nvme2tb/nachammai/FeatureExtraction/AVGtrROS/' + 'Fifth50AVG.trROS.npy', output.astype(np.float16))