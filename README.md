### Noise-Matrix

```bash

The model created in Subtask1 has the average precision accuracy of 
Top L/5 79.19 Top L 53.52 NC 50.07. So, in this sub task the intent is to calculate the Noise/Mean matrix. 
Every feature is in the shape of [L, L, X], where X denotes the dimension of the feature and L denotes the length of the protein chosen.
For e.g. X = 441 for the covariance/precision feature. To calculate the Noise/Mean matrix for the different features,
the top 200 out of 3456 proteins from Deepcov dataset was selected based on max L criteria and mean of those features were calculated.
Mean was calculated using the numpy method nanmean. Here we initialize the mean matrix with np.nan to overcome the ‘zero’s affecting the average problem’,
since every chosen protein (among the 200) will be of varying lengths, so when a value at a particular place in a sequence is zero,
this could create a bias in a particular place in a protein sequence. 
