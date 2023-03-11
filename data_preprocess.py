
## Import necessary libraries
import scipy.io as sio
import numpy as np
import tqdm


# Function to create patches around a groundtruth pixel from an image
# Function takes the hsi image, groundtruth image and windowsize
# as input and returns patches and correpsonding
# extracted grountruth label 
def patchmaker(X, y, windowSize=11):

  import tqdm
  shapeX = np.shape(X)

  margin = int((windowSize-1)/2)
  newX = np.empty([shapeX[0]+2*margin,shapeX[1]+2*margin,shapeX[2]])

  newX[margin:shapeX[0]+margin:,margin:shapeX[1]+margin,:] = X

  index = np.empty([0,3], dtype = 'int')

  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeX[0]):
      for j in range(shapeX[1]):
        if y[i,j] == k:
          index = np.append(index,np.expand_dims(np.array([k,i,j]),0),0)

  patchesX = np.empty([index.shape[0],2*margin+1,2*margin+1,shapeX[2]])
  patchesY = np.empty([index.shape[0]])

  for i in range(index.shape[0]):
    p = index[i,1]
    q = index[i,2]
    patchesX[i,:,:,:] = newX[p:p+windowSize,q:q+windowSize,:]
    patchesY[i] = index[i,0]

  return patchesX, patchesY-1


## Read imgagefile and groundtruth
data = sio.loadmat('data/Indian_pines_corrected.mat')['indian_pines_corrected']
label = sio.loadmat('data/Indian_pines_gt.mat')['indian_pines_gt']


# Normalize the data using min-max normalization
feats_norm = np.empty([145,145,200], dtype = 'float32')
for i in tqdm.tqdm(range(200)):
  feats_norm[:,:,i] = data[:,:,i] - np.min(data[:,:,i])
  feats_norm[:,:,i] = feats_norm[:,:,i]/np.max(feats_norm[:,:,i])


# Create unsupervised patches of size 11 x 11 for AE
patches = np.empty([18225,11,11,200], dtype = 'float32')
k = 0
for i in tqdm.tqdm(range(5,140)): 
  for j in range(5,140):
    patch = feats_norm[i-5:i+6,j-5:j+6,:]
    patches[k,:,:,:] = patch

# Create supervised patches of size 11 x 11 around groundtruth pixel
train_test_patches, train_test_labels = patchmaker(feats_norm, label, 11)


# Spilt the supervised patches into train and test sets 
# with 10% training samples
from sklearn.model_selection import StratifiedShuffleSplit
s3 = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=0)
s3.get_n_splits(train_test_patches, train_test_labels)

for train_index, test_index in s3.split(train_test_patches, train_test_labels):
   train_patches, test_patches = train_test_patches[train_index], train_test_patches[test_index]
   train_labels, test_labels = train_test_labels[train_index], train_test_labels[test_index]


## Save all unsupervised and supervised patches along with labels
np.save('data/train_patches',train_patches)
np.save('data/test_patches',test_patches)
np.save('data/train_labels',train_labels)
np.save('data/test_labels',test_labels)

np.save('data/patches_salinas',patches)
