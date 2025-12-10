import pickle
import numpy as np
X512 = np.full((50, 512, 512, 55), np.nan, dtype=np.float16)

def get_feature_55(pdb, features_path, expected_n_channels):
    features = pickle.load(open(features_path + pdb + '.pkl', 'rb'))
    l = len(features['seq'])
    seq = features['seq']
    # Create X and Y placeholders
    X = np.full((l, l, expected_n_channels), 0.0)
    # Add secondary structure
    ss = features['ss']
    assert ss.shape == (3, l)
    fi = 0
    for j in range(3):
        a = np.repeat(ss[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add PSSM
    pssm = features['pssm']
    assert pssm.shape == (21, l)
    for j in range(21):
        a = np.repeat(pssm[j].reshape(1, l), l, axis = 0)
        X[:, :, fi] = a
        fi += 1
        X[:, :, fi] = a.T
        fi += 1
    # Add SA
    sa = features['sa']
    assert sa.shape == (l, )
    a = np.repeat(sa.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add entrophy
    entropy = features['entropy']
    assert entropy.shape == (l, )
    a = np.repeat(entropy.reshape(1, l), l, axis = 0)
    X[:, :, fi] = a
    fi += 1
    X[:, :, fi] = a.T
    fi += 1
    # Add CCMpred
    ccmpred = features['ccmpred']
    assert ccmpred.shape == ((l, l))
    X[:, :, fi] = ccmpred
    fi += 1
    # Add  FreeContact
    freecon = features['freecon']
    assert freecon.shape == ((l, l))
    X[:, :, fi] = freecon
    fi += 1
    # Add potential
    potential = features['potential']
    assert potential.shape == ((l, l))
    X[:, :, fi] = potential
    fi += 1
    assert fi == expected_n_channels
    assert X.max() < 100.0
    assert X.min() > -100.0
    return X

list=[]
f=open('/nvme2tb/nachammai/FeatureExtraction/Fifth_50.lst','r')
for l in f.readlines():
    list.append(l.strip().split()[0])
    
fpath = '/backup/pdnet-feature-analysis/gen-feats/data/deepcov/features/'
fpathsfx = '.pkl'
a = 'rb'
for i in range(50):
    fd=[]
    fd=pickle.load(open(fpath + list[i] + fpathsfx,a))
    L=len(fd['seq'])

    x_pdnet = get_feature_55(list[i], fpath, 55)

    X512[i,:L,:L,:] = x_pdnet
    
output = np.nanmean(X512, axis=0)

np.save( '/nvme2tb/nachammai/FeatureExtraction/AVGpdnet55/' + 'Fifth50AVG.pdnet55.npy', output.astype(np.float16))