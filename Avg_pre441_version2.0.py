import numpy as np
X512 = np.full((50, 512, 512, 441), np.nan, dtype=np.float16)

list=[]
f=open('/hdd1tb/nachammai/FeatureExtraction/Fifth_50.lst','r')
for l in f.readlines():
    list.append(l.strip().split()[0])
    
fpath = '/hdd1tb/nachammai/ssd2tb/pre441/'
fpathsfx = '.pre441.npy'
for i in range(50):
    fd=[]
    fd=np.load(fpath + list[i] + fpathsfx)
    L=len(fd[0,:,0,0])
    X512[i,:L,:L,:] = fd
    
output = np.nanmean(X512, axis=0)

np.save( '/hdd1tb/nachammai/FeatureExtraction/AVGpre441/' + 'Fifth50AVG.pre441.npy', output.astype(np.float16))