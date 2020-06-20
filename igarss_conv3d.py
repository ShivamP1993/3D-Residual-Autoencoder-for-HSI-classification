# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose
from keras.layers import Reshape, Conv2DTranspose, Concatenate
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from sklearn.metrics import confusion_matrix

# Funtion to find accuracies. Input is the confusion matrix

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# Patches for unsupervised training of autoencoder
patches_hsi = np.load('/data/patches_hsi.npy')

# Reshaping patches as per input for the Conv3D 
patches_hsi = np.reshape(patches_hsi, [18225,11,11,200,1])

# Training patches for input in AE to get output with reduced dimensions. These can be used to train the classifier

train_patches = np.load('/data/train_patches.npy')
train_labels = np.load('/data/train_labels.npy')

# Test patches for input in AE to get output with reduced dimensions. These can be used to evaluate the classifier

test_patches = np.load('/data/test_patches.npy')
test_labels = np.load('/data/test_labels.npy')

# Reshaping train and test patches as per input for the Conv3D 

train_patches = np.reshape(train_patches, [1024,11,11,200,1])
test_patches = np.reshape(test_patches, [9225,11,11,200,1])

## 3D Residual Autoencoder model

K.clear_session()
g = tf.Graph()

k = 0
with g.as_default():

  x = Input(shape=(11,11,200,1), name='inputA')   # Input of AAE module

  # Encoder
  
  c1 = Conv3D(32, (3,3,4), strides=(1,1,1), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c1')(x)
  
  c2 = Conv3D(64, (3,3,3), strides=(1,1,1), padding='same', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c2')(c1)

  res1 = Concatenate(axis = 4)([c1,c2])   # Residual Connection 1

  c3 = Conv3D(128, (3,3,5), strides=(1,1,2), padding='valid', 
                            activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c3')(c2)
  
  c4 = Conv3D(64, (5,5,3), strides=(1,1,1), padding='same', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c4')(c3)

  res2 = Concatenate(axis = 4)([c3, c4]) # Residual Connection 2

  c5 = Conv3D(32, (5,5,5), strides=(1,1,2), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c5')(c4)

  
  # Encoded Features
  c_int = Conv3D(10, (3,3,5), strides=(1,1,1), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c_int')(c5)

  
  ## Decoder
  
  c6 = Conv3DTranspose(32, (5,5,7), strides=(1,1,2), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c6')(c_int)
  
  c7 = Conv3DTranspose(64, (5,5,5), strides=(1,1,1), padding='same',  
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c7')(c6)

  c8 = Conv3DTranspose(128, (3,3,7), strides=(1,1,2), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c8')(c7)

  c9 = Conv3DTranspose(64, (3,3,3), strides=(1,1,1), padding='same',
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c9')(c8)
  
  c10 = Conv3DTranspose(32, (3,3,8), strides=(1,1,1), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c10')(c9)
  
  c11 = Conv3DTranspose(1, (3,3,7), strides=(1,1,1), padding='valid', 
                             activation='relu', use_bias=True, kernel_initializer='glorot_uniform', name = 'c11')(c10)
  ae = Model([x], c11, name = 'ae')
  optim = keras.optimizers.Nadam(0.00005, beta_1=0.9, beta_2=0.999)
  ae.compile(loss='mean_squared_error', optimizer=optim)

  for epoch in range(200):
    ae.fit(x = patches_hsi, y = patches_hsi, epochs=1, batch_size = 64, verbose = 1)
    print('S_epoch = ', epoch)  
    ae.save('/models/ae.h5') # Saving after every epoch

## Loading saved model to get encoded features

K.clear_session()
g = tf.Graph()
with g.as_default():

  # Loading model
  ae2 = keras.models.load_model('/models/ae')
  
  layer_name = 'c_int' # Name of layer with encoded features 

  layer_model = keras.Model(ae2.input, ae2.get_layer(layer_name).output)

  train_feats = layer_model.predict(np.reshape(train_patches, [1024, 11,11,200,1])) # Encoded trained features
  train_feats = np.reshape(train_feats, [1024,430]) # Flattening encoded trained features

  test_feats = layer_model.predict(np.reshape(test_patches, [9225, 11,11,200,1])) # Encoded test features
  test_feats = np.reshape(test_feats, [9225,430]) # Flattening encoded test features

## Random Forest Classifier to train on encoded training samples and test on encoded test samples

from sklearn.ensemble import RandomForestClassifier

# Initialize RF classifier with 500 trees
rf = RandomForestClassifier(random_state=100, n_estimators = 500) 

# Fitting the classifier
rf.fit(train_feats, train_labels)

# Confusion Matrix
conf = confusion_matrix(rf.predict(test_feats), test_labels)

# Accuracy
ovr_acc, _, _, _, _ = accuracies(conf)  

#Print accuracy
print(np.round(100*ovr_acc,2))