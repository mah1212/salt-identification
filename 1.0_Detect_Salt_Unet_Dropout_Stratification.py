#Import some useful libraries
import numpy as np #array handling and linear algebra
import matplotlib.pyplot as plt #plotting lib


#Data loading
from PIL import Image
import cv2 as cv
import os
from pathlib import Path
import glob

import pandas as pd

from tqdm import tqdm_notebook

import keras

print(keras.__version__)

from keras.layers import Conv2D 
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
print("Importing finished!")


# =============================================================================
# 
# Data
# 
# The data is a set of images chosen at various locations chosen at random in the 
# subsurface. The images are 101 x 101 pixels and each pixel is classified as either 
# salt or sediment. In addition to the seismic images, the depth of the imaged location 
# is provided for each image. The goal of the competition is to segment regions that 
# contain salt.
# 
# 
# =============================================================================

#Machine learning framework
# =============================================================================
# ResNet
# 
# keras.applications.resnet.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet.ResNet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet.ResNet152(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet_v2.ResNet50V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet_v2.ResNet101V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnext.ResNeXt50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# keras.applications.resnext.ResNeXt101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
# 
# ResNet, ResNetV2, ResNeXt models, with weights pre-trained on ImageNet.
# 
# This model and can be built both with 'channels_first' data format (channels, height, width) or 'channels_last' data format (height, width, channels).
# 
# The default input size for this model is 224x224.
# Arguments
# 
#     include_top: whether to include the fully-connected layer at the top of the network.
#     weights: one of None (random initialization) or 'imagenet' (pre-training on ImageNet).
#     input_tensor: optional Keras tensor (i.e. output of layers.Input()) to use as image input for the model.
#     input_shape: optional shape tuple, only to be specified if include_top is False (otherwise the input shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with 'channels_first' data format). It should have exactly 3 inputs channels, and width and height should be no smaller than 32. E.g. (200, 200, 3) would be one valid value.
#     pooling: Optional pooling mode for feature extraction when include_top is False.
#         None means that the output of the model will be the 4D tensor output of the last convolutional layer.
#         'avg' means that global average pooling will be applied to the output of the last convolutional layer, and thus the output of the model will be a 2D tensor.
#         'max' means that global max pooling will be applied.
#     classes: optional number of classes to classify images into, only to be specified if include_top is True, and if no weights argument is specified.
# 
# Returns
# 
# A Keras Model instance.
# 
# =============================================================================

# =============================================================================
# 
# Typical input image sizes to a Convolutional Neural Network trained on ImageNet 
# are 224×224, 227×227, 256×256, and 299×299; however, you may see other dimensions 
# as well.
# 
# VGG16, VGG19, and ResNet all accept 224×224 input images while 
# Inception V3 and Xception require 299×299 pixel inputs
# 
# 
# =============================================================================

# =============================================================================
# 
# api reference r1.13
# tf.layers.conv2d(
#     inputs,
#     filters,
#     kernel_size,
#     strides=(1, 1),
#     padding='valid',
#     data_format='channels_last',
#     dilation_rate=(1, 1),
#     activation=None,
#     use_bias=True,
#     kernel_initializer=None,
#     bias_initializer=tf.zeros_initializer(),
#     kernel_regularizer=None,
#     bias_regularizer=None,
#     activity_regularizer=None,
#     kernel_constraint=None,
#     bias_constraint=None,
#     trainable=True,
#     name=None,
#     reuse=None
#     
#     
# )
# 
# 2D convolution layer (e.g. spatial convolution over images).
# 
# This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.
# 
# When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the batch axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".
# 
# Arguments
# 
#     filters: Integer, the dimensionality of the output space (i.e. the number of 
#               output filters in the convolution).
# 
#     kernel_size: An integer or tuple/list of 2 integers, specifying the height 
#                 and width of the 2D convolution window. Can be a single integer 
#                 to specify the same value for all spatial dimensions.
# 
#     strides: An integer or tuple/list of 2 integers, specifying the strides of 
#              the convolution along the height and width. Can be a single integer 
#              to specify the same value for all spatial dimensions. Specifying any 
#              stride value != 1 is incompatible with specifying any dilation_rate 
#              value != 1.
# 
#     padding: one of "valid" or "same" (case-insensitive). Note that "same" is 
#              slightly inconsistent across backends with strides != 1, as described 
#              here
# 
#     data_format: A string, one of "channels_last" or "channels_first". The ordering 
#                  of the dimensions in the inputs. "channels_last" corresponds to 
#                  inputs with shape (batch, height, width, channels) while 
#                  "channels_first" corresponds to inputs with shape (batch, 
#                 channels, height, width). It defaults to the image_data_format 
#                 value found in your Keras config file at ~/.keras/keras.json. 
#                 If you never set it, then it will be "channels_last".
# 
#     dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation 
#                    rate to use for dilated convolution. Can be a single integer 
#                    to specify the same value for all spatial dimensions. Currently, 
#                    specifying any dilation_rate value != 1 is incompatible with 
#                    specifying any stride value != 1.
# 
#     activation: Activation function to use (see activations). If you don't specify 
#                 anything, no activation is applied (ie. "linear" activation: 
#                     a(x) = x).
# 
#     use_bias: Boolean, whether the layer uses a bias vector.
# 
#     kernel_initializer: Initializer for the kernel weights matrix 
#                         (see initializers).
# 
#     bias_initializer: Initializer for the bias vector (see initializers).
# 
#     kernel_regularizer: Regularizer function applied to the kernel weights matrix 
#                         (see regularizer).
# 
#     bias_regularizer: Regularizer function applied to the bias vector 
#                      (see regularizer).
# 
#     activity_regularizer: Regularizer function applied to the output of the layer 
#                           (its "activation"). (see regularizer).
# 
#     kernel_constraint: Constraint function applied to the kernel matrix 
#                        (see constraints).
# 
#     bias_constraint: Constraint function applied to the bias vector 
#                      (see constraints).
# 
# Input shape
# 
# 4D tensor with shape: (batch, channels, rows, cols) if data_format is "channels_first" or 4D tensor with shape: (batch, rows, cols, channels) if data_format is "channels_last".
# 
# Output shape
# 
# 4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format is "channels_first" or 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format is "channels_last". rows and cols values might have changed due to padding.
# 
# 
# =============================================================================



''' We are inside cell_images folder'''
# Load Data using os
# Print current directory
print(os.getcwd())

# Print all the directories in the current directory
print(os.listdir())

''' File path reading technique NO MORE OS.Path '''
# Read and save current directory path using Path()
data_dir = Path('..\\Salt')
print(data_dir)



# Read train test directory

train_dir = data_dir / 'train'
print(train_dir)

test_dir = data_dir / 'test'
print(test_dir)



# ----- Model Versioning ----
version = 1
model_name = f'unet_resnet_v{version}'
saved_model = model_name + '.model'
submission_file = model_name + '.csv'



#--- Load train & depth data ---
# train.csv A helper file that shows the training set masks 
# in run-length encoded format
train_df_normal = pd.read_csv('train.csv')
train_df_normal.head()

train_df = pd.read_csv('train.csv', index_col='id', usecols=[0])
train_df.head()
print(len(train_df)) # 4000
print(train_df.shape) # (4000, 0)
print(train_df.describe())



# The depth underground (in feet) of each image 
depths_df_normal = pd.read_csv('depths.csv')
depths_df_normal.head()

depths_df = pd.read_csv('depths.csv', index_col='id')
depths_df.head()
print(len(depths_df))
print(depths_df.shape) # (22000, 0)


# check for commoon id
common_idx = train_df.index.intersection(depths_df.index)
print (common_idx)

train_df.loc[common_idx].head()
depths_df.loc[common_idx].head()


#---- Image upsample / downsample ---
from skimage.transform import resize
img_size_original = 101
img_size_target = 128

def upsample(img):# not used
    if img_size_original == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    
def downsample(img):# not used
    if img_size_original == img_size_target:
        return img
    return resize(img, (img_size_original, img_size_original), mode='constant', preserve_range=True)





#---- Create test data --------
# Pandas join/merge https://www.shanelynn.ie/merge-join-dataframes-python-pandas-index-1/
# https://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.join.html
# https://chrisalbon.com/python/data_wrangling/pandas_join_merge_dataframe/    
train_df = train_df.join(depths_df)
train_df.shape # (4000, 1)
train_df.head()

print(len(~depths_df)) # 22000
print(~depths_df.shape) # shows error

print(~depths_df.index.isin(train_df.index))
depths_df[~depths_df.index.isin(train_df.index)]

test_df = depths_df[~depths_df.index.isin(train_df.index)]
test_df.shape # (18000, 1)
test_df.head()


import tensorflow as tf
from keras.preprocessing.image import img_to_array, array_to_img, load_img





#------ Load image into train df -----
train_df.shape # (4000, 1)
train_df.head()

# Read train image, train mask
train_img_dir = train_dir / 'images'
print(train_img_dir)





# Load image, convert image to grayscale, normalize image, convert image to numpy array

train_df['images'] = [np.array(load_img('{}\\{}.png'.format(train_img_dir, idx), 
                    color_mode = 'grayscale'
                    )) / 255 for idx in train_df.index]


train_df.head()
train_df.shape # (4000, 2)


# Add a new column for masks, load image, convert image, normalize image.
train_df["img_masks"] = [np.array(load_img("..\\Salt\\train\\masks\\{}.png".format(idx), 
                    color_mode = 'grayscale')) / 255 for idx in tqdm_notebook(train_df.index)]

train_df.head()
train_df.shape # (4000, 3)




# Numpy Sum https://www.sharpsightlabs.com/blog/numpy-sum/
# Pandas map function https://www.youtube.com/watch?v=yuNbn9cczjA

'''

Calculating the salt coverage and salt coverage classes

Counting the number of salt pixels in the masks and dividing them by the image 
size. Also create 11 coverage classes, -0.1 having no salt at all to 1.0 being 
salt only. Plotting the distribution of coverages and coverage classes, and the 
class against the raw coverage.

'''

#train_df.img_masks.map(np.sum)

#print(pow(121, 2))


train_df['coverage'] = train_df.img_masks.map(np.sum) / pow(img_size_original, 2)


def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i

#print(cov_to_class(9))

train_df['coverage_class'] = train_df.coverage.map(cov_to_class)
train_df.head()


import seaborn as sns
fig, axs = plt.subplots(1, 2, figsize=(10,3))
sns.distplot(train_df.coverage, ax=axs[0], kde = False)
sns.distplot(train_df.coverage_class, ax=axs[1], bins=10)
plt.suptitle("Salt Coverage")
axs[0].set_xlabel['Salt Coverage']
axs[1].set_xlabel['Salt Coverage Class']


# plot depth distribution
sns.distplot(train_df.z, label="Train")
sns.distplot(test_df.z, label="Test")
plt.legend()
plt.title("Depth Distribution")


train_df.shape
print(len(train_df.index[:48]))

# Show image, mask
max_images = 60

grid_width = 12

grid_height = int(max_images / grid_width) # 4

fig, axs = plt.subplots(grid_height, grid_width, figsize=(grid_width, grid_height))

for i, idx in enumerate(train_df.index[:max_images]):
    
    img = train_df.loc[idx].images
    img_mask = train_df.loc[idx].img_masks
    
    
    ax = axs[int(i / grid_width), i % grid_width]
    
    
    ax.imshow(img, cmap='Greys')
    ax.imshow(img_mask, alpha=0.3, cmap='Greens')

    # print depth
    ax.text(1, img_size_original-1, train_df.loc[idx].z, color='Black')
    
    # print coverage
    ax.text(img_size_original - 1, 1, round(train_df.loc[idx].coverage, 2), 
            ha='right', va='top', color='Black' )
    
    # print coverage class
    ax.text(1, 1, train_df.loc[idx].coverage_class, ha='left', va='top', 
            color='Black')
    
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
plt.suptitle('Green: Salt. Top-left: coverage_class, Top-right: coverage, Bottom-left: depth')




#------------------ Train Test Split ----------------------
'''
If the number of values belonging to each class are unbalanced, using stratified 
sampling is a good thing. You are basically asking the model to take the training 
and test set such that the class proportion is same as of the whole dataset, 
which is the right thing to do. If your classes are balanced then a shuffle 
(no stratification needed here) can basically guarantee a fair test and train split.

Now your model will be capable or at least enough equipped to predict the 
outnumbered class (class with lesser points in number). That is why instead 
of just calculating Accuracy, you have been given other metrics like 
Sensitivity and Specificity. Keep a watch on these, these are the guardians.

Hope this helps.

'''
from sklearn.model_selection import train_test_split

ids_train, ids_valid, x_train, x_valid, y_train, y_valid, conv_train, conv_test, depth_train, depth_test = train_test_split(
        
        train_df.index.values,
        np.array(train_df.images.map(upsample).tolist()).reshape(
                -1, img_size_target, img_size_target, 1), 
        np.array(train_df.img_masks.map(upsample).tolist()).reshape(
                -1, img_size_target, img_size_target, 1), 
        train_df.coverage.values, 
        train_df.z.values,
        test_size = 0.2,
        
        stratify = train_df.coverage_class,
        random_state = 1203 
        )
        

tmp_img = np.zeros((img_size_target, img_size_target), dtype=train_df.images.loc[ids_train[10]].dtype)

print(np.zeros((img_size_target, img_size_target)))
print(np.zeros((img_size_target, img_size_target), dtype=train_df.images.loc[ids_train[10]].dtype))


tmp_img[:img_size_original, :img_size_original] = train_df.images.loc[ids_train[10]]
fix, axs = plt.subplots(1, 2, figsize=(10,5))
axs[0].imshow(tmp_img, cmap="Greys")
axs[0].set_title("Original image")
axs[1].imshow(x_train[10].squeeze(), cmap="Greys")
axs[1].set_title("Scaled image")



# =============================================================================
# 
# Unet Network Architecture
# 
# The network architecture is illustrated in Figure 1. It consists of a contracting
# path (left side) and an expansive path (right side). The contracting path follows
# the typical architecture of a convolutional network. It consists of the repeated
# application of two 3x3 convolutions (unpadded convolutions), each followed by
# a recti
ed linear unit (ReLU) and a 2x2 max pooling operation with stride 2
# for downsampling. At each downsampling step we double the number of feature
# channels. Every step in the expansive path consists of an upsampling of the
# feature map followed by a 2x2 convolution (\up-convolution") that halves the
# number of feature channels, a concatenation with the correspondingly cropped
# feature map from the contracting path, and two 3x3 convolutions, each fol-
# lowed by a ReLU. The cropping is necessary due to the loss of border pixels in
# every convolution. At the 
nal layer a 1x1 convolution is used to map each 64-
# component feature vector to the desired number of classes. In total the network
# has 23 convolutional layers.
# To allow a seamless tiling of the output segmentation map (see Figure 2), it
# is important to select the input tile size such that all 2x2 max-pooling operations
# are applied to a layer with an even x- and y-size.
# 
# 
# =============================================================================


# =============================================================================
# 
# 3.1 Data Augmentation
# 
# Data augmentation is essential to teach the network the desired invariance and
# robustness properties, when only few training samples are available. In case of
# 6
# microscopical images we primarily need shift and rotation invariance as well as
# robustness to deformations and gray value variations. Especially random elas-
# tic deformations of the training samples seem to be the key concept to train
# a segmentation network with very few annotated images. We generate smooth
# deformations using random displacement vectors on a coarse 3 by 3 grid. The
# displacements are sampled from a Gaussian distribution with 10 pixels standard
# deviation. Per-pixel displacements are then computed using bicubic interpola-
# tion. Drop-out layers at the end of the contracting path perform further implicit
# data augmentation.
# 
# =============================================================================


# -------------- Build Unet Model ------------------------------
def build_model(input_layer, start_neurons):
    
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2,2))(conv1)
    pool1 = Dropout(.25)(pool1)


    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2,2))(conv2)
    pool2 = Dropout(.2)(pool2)
    
    
    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2,2))(conv3)
    pool3 = Dropout(.25)(pool3)


    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D((2,2))(conv4)
    pool4 = Dropout(.2)(pool4)


    # Middle
    convm = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(pool4)
    convm = Conv2D(start_neurons * 16, (3, 3), activation='relu', padding='same')(convm)
   
    
    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides = (2, 2), padding='same')(convm)
    
    uconv4 = concatenate([deconv4, conv4]) # conv4
    
    uconv4 = Dropout(.25)(uconv4)
    
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4) # no strides
    uconv4 = Conv2D(start_neurons * 8, (3, 3), activation='relu', padding='same')(uconv4)
    
    
    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides = (2, 2), padding='same')(uconv4)    
    
    uconv3 = concatenate([deconv3, conv3]) # conv3 <----- Important!
    
    uconv3 = Dropout(.25)(uconv3)
    
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation='relu', padding='same')(uconv3)
    


    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (3,3), strides=(2,2), padding='same')(uconv3)

    uconv2 = concatenate([deconv2, conv2])
    
    uconv2 = Dropout(.25)(uconv2)
    
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation='relu', padding='same')(uconv2)


    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2,2), padding='same')(uconv2)
    
    uconv1 = concatenate([deconv1, conv1])
    
    uconv1 = Dropout(.25)(uconv1)
    
    uconv1 = Conv2D(start_neurons * 1, (3,3), activation='relu', padding='same')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3,3), activation='relu', padding='same')(uconv1)


    # Output layer
    output_layer = Conv2D(1, (1,1), activation='sigmoid')(uconv1)
    
    return output_layer


input_layer = Input(shape=(img_size_target, img_size_target, 1))

#input_layer = Input(shape=(128, 128, 1))
output_layer = build_model(input_layer, 16)

model = Model(input_layer, output_layer)
model.summary()

# =============================================================================
# 
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 128, 128, 1)  0                                            
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 128, 128, 16) 160         input_1[0][0]                    
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 128, 128, 16) 2320        conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_1 (MaxPooling2D)  (None, 64, 64, 16)   0           conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 64, 64, 16)   0           max_pooling2d_1[0][0]            
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 64, 64, 32)   4640        dropout_1[0][0]                  
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 64, 64, 32)   9248        conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_2 (MaxPooling2D)  (None, 32, 32, 32)   0           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 32, 32, 32)   0           max_pooling2d_2[0][0]            
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 32, 32, 64)   18496       dropout_2[0][0]                  
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 32, 32, 64)   36928       conv2d_5[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_3 (MaxPooling2D)  (None, 16, 16, 64)   0           conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# dropout_3 (Dropout)             (None, 16, 16, 64)   0           max_pooling2d_3[0][0]            
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 16, 16, 128)  73856       dropout_3[0][0]                  
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 16, 16, 128)  147584      conv2d_7[0][0]                   
# __________________________________________________________________________________________________
# max_pooling2d_4 (MaxPooling2D)  (None, 8, 8, 128)    0           conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 8, 8, 128)    0           max_pooling2d_4[0][0]            
# __________________________________________________________________________________________________
# conv2d_9 (Conv2D)               (None, 8, 8, 256)    295168      dropout_4[0][0]                  
# __________________________________________________________________________________________________
# conv2d_10 (Conv2D)              (None, 8, 8, 256)    590080      conv2d_9[0][0]                   
# __________________________________________________________________________________________________
# conv2d_transpose_1 (Conv2DTrans (None, 16, 16, 128)  295040      conv2d_10[0][0]                  
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 16, 16, 256)  0           conv2d_transpose_1[0][0]         
#                                                                  conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# dropout_5 (Dropout)             (None, 16, 16, 256)  0           concatenate_1[0][0]              
# __________________________________________________________________________________________________
# conv2d_11 (Conv2D)              (None, 16, 16, 128)  295040      dropout_5[0][0]                  
# __________________________________________________________________________________________________
# conv2d_12 (Conv2D)              (None, 16, 16, 128)  147584      conv2d_11[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_2 (Conv2DTrans (None, 32, 32, 64)   73792       conv2d_12[0][0]                  
# __________________________________________________________________________________________________
# concatenate_2 (Concatenate)     (None, 32, 32, 128)  0           conv2d_transpose_2[0][0]         
#                                                                  conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# dropout_6 (Dropout)             (None, 32, 32, 128)  0           concatenate_2[0][0]              
# __________________________________________________________________________________________________
# conv2d_13 (Conv2D)              (None, 32, 32, 64)   73792       dropout_6[0][0]                  
# __________________________________________________________________________________________________
# conv2d_14 (Conv2D)              (None, 32, 32, 64)   36928       conv2d_13[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_3 (Conv2DTrans (None, 64, 64, 32)   18464       conv2d_14[0][0]                  
# __________________________________________________________________________________________________
# concatenate_3 (Concatenate)     (None, 64, 64, 64)   0           conv2d_transpose_3[0][0]         
#                                                                  conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# dropout_7 (Dropout)             (None, 64, 64, 64)   0           concatenate_3[0][0]              
# __________________________________________________________________________________________________
# conv2d_15 (Conv2D)              (None, 64, 64, 32)   18464       dropout_7[0][0]                  
# __________________________________________________________________________________________________
# conv2d_16 (Conv2D)              (None, 64, 64, 32)   9248        conv2d_15[0][0]                  
# __________________________________________________________________________________________________
# conv2d_transpose_4 (Conv2DTrans (None, 128, 128, 16) 4624        conv2d_16[0][0]                  
# __________________________________________________________________________________________________
# concatenate_4 (Concatenate)     (None, 128, 128, 32) 0           conv2d_transpose_4[0][0]         
#                                                                  conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# dropout_8 (Dropout)             (None, 128, 128, 32) 0           concatenate_4[0][0]              
# __________________________________________________________________________________________________
# conv2d_17 (Conv2D)              (None, 128, 128, 16) 4624        dropout_8[0][0]                  
# __________________________________________________________________________________________________
# conv2d_18 (Conv2D)              (None, 128, 128, 16) 2320        conv2d_17[0][0]                  
# __________________________________________________________________________________________________
# conv2d_19 (Conv2D)              (None, 128, 128, 1)  17          conv2d_18[0][0]                  
# ==================================================================================================
# Total params: 2,158,417
# Trainable params: 2,158,417
# Non-trainable params: 0
# __________________________________________________________________________________________________
# 
# 
# =============================================================================

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary() # same as above




# Shuffle the data

''' How to shuffle pandas dataframe?
The more idiomatic way to do this with pandas is to use the .sample method of 
your dataframe, i.e.

df.sample(frac=1)

The frac keyword argument specifies the fraction of rows to return in the random sample, 
so frac=1 means return all rows (in random order).

Note: If you wish to shuffle your dataframe in-place and reset the index, 
you could do e.g.

df = df.sample(frac=1).reset_index(drop=True)

Here, specifying drop=True prevents .reset_index from creating a column containing 
the old index entries.
'''
train_df = train_df.sample(frac=1).reset_index(drop=True)
train_df.head()
train_df.tail()


''' How many samples for each class are there? '''
class_count = train_df['label'].value_counts()
print(class_count)


import seaborn as sns

plt.figure(figsize=(10,8))
sns.barplot(x = class_count.index, y = class_count.values)
plt.title('Normal vs Pneumonia')
plt.xlabel('Class type')
plt.ylabel('Count')
plt.xticks(range(len(class_count.index)), ['Normal(0)', 'Pneumonia(1)'])
plt.show()


''' Is Data balanced? Imbalanced? '''
# We can see from the plot that data is highly imbalanced

# plot images 
# Get samples from both classes, take 5 samples from each, total 10 samples
normal_samples = train_df[train_df['label']==0]['image'].iloc[:5].tolist()
pneumonia_samples = train_df[train_df['label']==1]['image'].iloc[:5].tolist()

# Concatenate the samples in a single list and delete the above two lists
samples = normal_samples + pneumonia_samples 
del normal_samples, pneumonia_samples

# plot the samples
from skimage.io import imread

print(1//5)
print(1%5)

f, ax = plt.subplots(2,5, figsize=(30,10))
for i in range(10):
    img = imread(samples[i])
    ax[i//5, i%5].imshow(img, cmap='gray')
    if i<5:
        ax[i//5, i%5].set_title("Normal")
    else:
        ax[i//5, i%5].set_title("Pneumonia")
    ax[i//5, i%5].axis('off')
    ax[i//5, i%5].set_aspect('auto')
plt.show()



''' Prepare validation data '''
#. read directory
normal_dir = val_dir / 'NORMAL'
print(normal_dir)

pneumonia_dir = val_dir / 'PNEUMONIA'
print(pneumonia_dir)

# get the list
normal_img_list = normal_dir.glob('*.jpeg')
pneumonia_img_list = pneumonia_dir.glob('*.jpeg')

valid_data = []
valid_labels = []

# Some images are in grayscale while majority of them contains 3 channels. 
# So, if the image is grayscale, we will convert into a image with 3 channels.
# We will normalize the pixel values and resizing all the images to 128x128

# =============================================================================
# 
# to_categorical
# 
# keras.utils.to_categorical(y, num_classes=None, dtype='float32')
# 
# Converts a class vector (integers) to binary class matrix.
# 
# E.g. for use with categorical_crossentropy.
# 
# Arguments
# 
#     y: class vector to be converted into a matrix (integers from 0 to num_classes).
#     num_classes: total number of classes.
#     dtype: The data type expected by the input, as a string (float32, float64, int32...)
# 
# Returns
# 
# A binary matrix representation of the input. The classes axis is placed last.
# 
# Example
# 
# # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
# > labels
# array([0, 2, 1, 2, 0])
# # `to_categorical` converts this into a matrix with as many
# # columns as there are classes. The number of rows
# # stays the same.
# > to_categorical(labels)
# array([[ 1.,  0.,  0.],
#        [ 0.,  0.,  1.],
#        [ 0.,  1.,  0.],
#        [ 0.,  0.,  1.],
#        [ 1.,  0.,  0.]], dtype=float32)
# 
#     
# =============================================================================

from keras.utils import to_categorical

for img in normal_img_list:
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
        
    # Convert to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Scale
    img = img.astype(np.float32)/255.0
    
    # label
    label = to_categorical(0, num_classes = 2)
    
    # append
    valid_data.append(img)
    valid_labels.append(label)
    
    
    
for img in pneumonia_img_list:    
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
        
    # Convert to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Scale
    img = img.astype(np.float32)/255.0
    
    # label
    label = to_categorical(1, num_classes = 2)
    
    # append
    valid_data.append(img)
    valid_labels.append(label)

    
# Convert the image list into numpy array
valid_data = np.array(valid_data)    
valid_labels = np.array(valid_labels)

print('Valid Data Shape: ', valid_data.shape)
print('Valid Labels Shape: ', valid_labels.shape)
    
''' How to convert grayscale image to colored image? 
2. Why use np.dstack? 
3. Why convert to numpy array later?
4. How to check all the image channels at once? 
meaning if they are gray or colored?


'''


''' Create a data generator 
    Why?
    We will use keras functional api which uses model.fit_generator
    this model.fit_generator uses data_generator
    
    How to build a custom data generator?
    
    keras.preprocessing.image.
    ImageDataGenerator(featurewise_center=False, 
                       samplewise_center=False, 
                       featurewise_std_normalization=False, 
                       samplewise_std_normalization=False, 
                       zca_whitening=False, 
                       zca_epsilon=1e-06, 
                       rotation_range=0, 
                       width_shift_range=0.0, 
                       height_shift_range=0.0, 
                       brightness_range=None, 
                       shear_range=0.0, 
                       zoom_range=0.0, 
                       channel_shift_range=0.0, 
                       fill_mode='nearest', 
                       cval=0.0, 
                       horizontal_flip=False, 
                       vertical_flip=False, 
                       rescale=None, 
                       preprocessing_function=None, 
                       data_format=None, 
                       validation_split=0.0, dtype=None)

'''

def data_generator(data, batch_size):
    
    # Get total number of samples in the data
    n = len(data)
    steps = n/batch_size
    
    # Define 2 numpy arrays for containing batch data and batch labels
    batch_data = np.zeros((batch_size, 128, 128, 3), dtype = np.float32)
    batch_labels = np.zeros((batch_size, 2), dtype = np.float32)
    
    # Get a numpy array for all the indices of the input data
    indices = np.arange(n)
    
    
    i = 0
    while True:
        
        # shuffle indices
        np.random.shuffle(indices)
        
        # Get the next batch
        next_batch = indices[(i*batch_size):(i+1)*batch_size]
        
        count = 0
        for j, idx in enumerate(next_batch):
            img_path = data.iloc[idx]['image']
            label = data.iloc[idx]['label']
            
            # one hot encoding
            encoded_label = to_categorical(label, num_classes = 2)
            
            
            # read the image and resize
            img = cv.imread(str(img_path))
            img = cv.resize(img, (128, 128))
            
            
            # check if it's a gray scale
            if img.shape[2] == 1:
                img = np.dstack(img, img, img)
                
            
            # convert BGR to RGB
            img_original = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            
            
            # normalize / scale the image
            img_original = img.astype(np.float32)/255.0
            
            
            # Store in the batch
            batch_data[count] = img_original
            batch_labels[count] = encoded_label
            
            if count==batch_size-1:
                break

        i+=1
        yield batch_data, batch_labels
            
        if i>=steps:
            i=0
            
            
            
# Create depthwise xception convolution neural network
# following this paper https://arxiv.org/abs/1610.02357
# keras link: https://keras.io/applications/#xception  
import keras
from keras.models import Sequential 
from keras.layers import Conv2D 
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import SeparableConv2D
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.models import Model

def build_model():
    input_img = Input(shape=(128, 128, 3), name='InputImage')
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2,2), name='Pool1')(x)
            

    x = SeparableConv2D(128, (3, 3,), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3, 3,), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2,2), name='Pool2')(x)    

    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = SeparableConv2D(256, (3,3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2,2), name='pool3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    
    x = SeparableConv2D(512, (3,3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2,2), name='pool4')(x)


    x = Dropout(0.2, name='dropout1')(x)
    x = Flatten(name='flatten')(x)
    
    x = Dense(units = 128, activation = "relu", name='Out1')(x)
    x = Dense(units = 2, activation = "softmax", name='Out2')(x)
    
  
    model = Model(inputs = input_img, outputs = x)
    return model



model = build_model()
model.summary()


''' Initialize with pre trained model
The default value of include_top parameter in VGG16 function is True. 
This means if you want to use a full layer pre-trained VGG network 
(with fully connected parts) you need to download 
vgg16_weights_tf_dim_ordering_tf_kernels.h5 file, 

If you want to use custom, download 
vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5

C:\Users\My_User_Name\.keras\datasets.
C:\Users\My_User_Name\.keras\model.

Then import the pre-trained MobileNet model. The Mobilenet (trained on the 
imagenet dataset for a thousand classes) will have a last layer consisting of 
1000 neurons (one for each class). We want as many neurons in the last layer 
of the network as the number of classes we wish to identify. So we discard the 
1000 neuron layer and add our own last layer for the network.

This can be done by setting (IncludeTop=False) when importing the model.

So suppose you want to train a dog breed classifier to identify 120 different 
breeds, we need 120 neurons in the final layer. This can be done using the 
following code.
'''
import keras

base_model = keras.applications.vgg16.VGG16(include_top=False, 
                                            weights='imagenet', 
                                            input_tensor=None, 
                                            input_shape=None)

base_model.summary()

# =============================================================================
# base_model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         (None, None, None, 3)     0         
# _________________________________________________________________
# block1_conv1 (Conv2D)        (None, None, None, 64)    1792      
# _________________________________________________________________
# block1_conv2 (Conv2D)        (None, None, None, 64)    36928     
# _________________________________________________________________
# block1_pool (MaxPooling2D)   (None, None, None, 64)    0         
# _________________________________________________________________
# block2_conv1 (Conv2D)        (None, None, None, 128)   73856     
# _________________________________________________________________
# block2_conv2 (Conv2D)        (None, None, None, 128)   147584    
# _________________________________________________________________
# block2_pool (MaxPooling2D)   (None, None, None, 128)   0         
# _________________________________________________________________
# block3_conv1 (Conv2D)        (None, None, None, 256)   295168    
# _________________________________________________________________
# block3_conv2 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_conv3 (Conv2D)        (None, None, None, 256)   590080    
# _________________________________________________________________
# block3_pool (MaxPooling2D)   (None, None, None, 256)   0         
# _________________________________________________________________
# block4_conv1 (Conv2D)        (None, None, None, 512)   1180160   
# _________________________________________________________________
# block4_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_conv3 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block4_pool (MaxPooling2D)   (None, None, None, 512)   0         
# _________________________________________________________________
# block5_conv1 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_conv2 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_conv3 (Conv2D)        (None, None, None, 512)   2359808   
# _________________________________________________________________
# block5_pool (MaxPooling2D)   (None, None, None, 512)   0         
# =================================================================
# Total params: 14,714,688
# Trainable params: 14,714,688
# Non-trainable params: 0
# 
# =============================================================================

# read only layer names
# =============================================================================
# for i,layer in enumerate(base_model.layers):
#     print(i,layer.name)
#     
# 0 input_1
# 1 block1_conv1
# 2 block1_conv2
# 3 block1_pool
# 4 block2_conv1
# 5 block2_conv2
# 6 block2_pool
# 7 block3_conv1
# 8 block3_conv2
# 9 block3_conv3
# 10 block3_pool
# 11 block4_conv1
# 12 block4_conv2
# 13 block4_conv3
# 14 block4_pool
# 15 block5_conv1
# 16 block5_conv2
# 17 block5_conv3
# 18 block5_pool    
# 
# =============================================================================

# =============================================================================
# # Get weights list
# for layer in model.layers:
#     weights = layer.get_weights()
#     print(weights)
# 
# =============================================================================


from __future__ import print_function

import h5py

def print_structure(weight_file_path):
    """
    Prints out the structure of HDF5 file.

    Args:
      weight_file_path (str) : Path to the file to analyze
    """
    f = h5py.File(weight_file_path)
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))

        if len(f.items())==0:
            return 

        for layer, g in f.items():
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items():
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for p_name in g.keys():
                param = g[p_name]
                print("      {}: {}".format(p_name, param.shape))
    finally:
        f.close()

weight_file_path= '..\\chest_xray\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
      
print_structure(weight_file_path)

# =============================================================================
# print_structure(weight_file_path)
# ..\chest_xray\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 contains: 
# Root attributes:
#   layer_names: [b'block1_conv1' b'block1_conv2' b'block1_pool' b'block2_conv1'
#  b'block2_conv2' b'block2_pool' b'block3_conv1' b'block3_conv2'
#  b'block3_conv3' b'block3_pool' b'block4_conv1' b'block4_conv2'
#  b'block4_conv3' b'block4_pool' b'block5_conv1' b'block5_conv2'
#  b'block5_conv3' b'block5_pool']
#   block1_conv1
#     Attributes:
#       weight_names: [b'block1_conv1_W_1:0' b'block1_conv1_b_1:0']
#     Dataset:
#       block1_conv1_W_1:0: (3, 3, 3, 64)
#       block1_conv1_b_1:0: (64,)
#   block1_conv2
#     Attributes:
#       weight_names: [b'block1_conv2_W_1:0' b'block1_conv2_b_1:0']
#     Dataset:
#       block1_conv2_W_1:0: (3, 3, 64, 64)
#       block1_conv2_b_1:0: (64,)
#   block1_pool
#     Attributes:
#       weight_names: []
#     Dataset:
#   block2_conv1
#     Attributes:
#       weight_names: [b'block2_conv1_W_1:0' b'block2_conv1_b_1:0']
#     Dataset:
#       block2_conv1_W_1:0: (3, 3, 64, 128)
#       block2_conv1_b_1:0: (128,)
#   block2_conv2
#     Attributes:
#       weight_names: [b'block2_conv2_W_1:0' b'block2_conv2_b_1:0']
#     Dataset:
#       block2_conv2_W_1:0: (3, 3, 128, 128)
#       block2_conv2_b_1:0: (128,)
#   block2_pool
#     Attributes:
#       weight_names: []
#     Dataset:
#   block3_conv1
#     Attributes:
#       weight_names: [b'block3_conv1_W_1:0' b'block3_conv1_b_1:0']
#     Dataset:
#       block3_conv1_W_1:0: (3, 3, 128, 256)
#       block3_conv1_b_1:0: (256,)
#   block3_conv2
#     Attributes:
#       weight_names: [b'block3_conv2_W_1:0' b'block3_conv2_b_1:0']
#     Dataset:
#       block3_conv2_W_1:0: (3, 3, 256, 256)
#       block3_conv2_b_1:0: (256,)
#   block3_conv3
#     Attributes:
#       weight_names: [b'block3_conv3_W_1:0' b'block3_conv3_b_1:0']
#     Dataset:
#       block3_conv3_W_1:0: (3, 3, 256, 256)
#       block3_conv3_b_1:0: (256,)
#   block3_pool
#     Attributes:
#       weight_names: []
#     Dataset:
#   block4_conv1
#     Attributes:
#       weight_names: [b'block4_conv1_W_1:0' b'block4_conv1_b_1:0']
#     Dataset:
#       block4_conv1_W_1:0: (3, 3, 256, 512)
#       block4_conv1_b_1:0: (512,)
#   block4_conv2
#     Attributes:
#       weight_names: [b'block4_conv2_W_1:0' b'block4_conv2_b_1:0']
#     Dataset:
#       block4_conv2_W_1:0: (3, 3, 512, 512)
#       block4_conv2_b_1:0: (512,)
#   block4_conv3
#     Attributes:
#       weight_names: [b'block4_conv3_W_1:0' b'block4_conv3_b_1:0']
#     Dataset:
#       block4_conv3_W_1:0: (3, 3, 512, 512)
#       block4_conv3_b_1:0: (512,)
#   block4_pool
#     Attributes:
#       weight_names: []
#     Dataset:
#   block5_conv1
#     Attributes:
#       weight_names: [b'block5_conv1_W_1:0' b'block5_conv1_b_1:0']
#     Dataset:
#       block5_conv1_W_1:0: (3, 3, 512, 512)
#       block5_conv1_b_1:0: (512,)
#   block5_conv2
#     Attributes:
#       weight_names: [b'block5_conv2_W_1:0' b'block5_conv2_b_1:0']
#     Dataset:
#       block5_conv2_W_1:0: (3, 3, 512, 512)
#       block5_conv2_b_1:0: (512,)
#   block5_conv3
#     Attributes:
#       weight_names: [b'block5_conv3_W_1:0' b'block5_conv3_b_1:0']
#     Dataset:
#       block5_conv3_W_1:0: (3, 3, 512, 512)
#       block5_conv3_b_1:0: (512,)
#   block5_pool
#     Attributes:
#       weight_names: []
#     Dataset:
# 
# =============================================================================
# =============================================================================
# 
# @mthrok I tried your function, it report
# print(" {}: {}".format(p_name, param.shape))
# 
# AttributeError: 'Group' object has no attribute 'shape'
# 
# However, when I tried print(" {}: {}".format(p_name, param.shape)) independently, 
# it's able to work. Do you have any idea about that?
# 
# Changing from param.shape to param on the line which failed seems to work in eliminating the AttributeError ronzilllia mentions.
# =============================================================================


# =============================================================================
# For keras 2
# from __future__ import print_function
# 
# import h5py
# 
# def print_structure(weight_file_path):
#     """
#     Prints out the structure of HDF5 file.
# 
#     Args:
#       weight_file_path (str) : Path to the file to analyze
#     """
#     f = h5py.File(weight_file_path)
#     try:
#         if len(f.attrs.items()):
#             print("{} contains: ".format(weight_file_path))
#             print("Root attributes:")
#         for key, value in f.attrs.items():
#             print("  {}: {}".format(key, value))
# 
#         if len(f.items())==0:
#             return 
# 
#         for layer, g in f.items():
#             print("  {}".format(layer))
#             print("    Attributes:")
#             for key, value in g.attrs.items():
#                 print("      {}: {}".format(key, value))
# 
#             print("    Dataset:")
#             for p_name in g.keys():
#                 param = g[p_name]
#                 subkeys = param.keys()
#                 for k_name in param.keys():
#                     print("      {}/{}: {}".format(p_name, k_name, param.get(k_name)[:]))
#     finally:
#         f.close()
# 
# 
# print_structure(weight_file_path)
# 
# =============================================================================




import h5py
# Open the VGG16 weight file
# f = file
f = h5py.File('..\\chest_xray\\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 'r')

# Select the layers for which you want to set weight.
# w = weight
# b = block

# w = f['block1_conv1']['block1_conv1_W_1:0']
# b = f['block1_conv1']['block1_conv1_b_1:0']

# in one line below: 
# w,b = f['block1_conv1']['block1_conv1_W_1:0'], f['block1_conv1']['block1_conv1_b_1:0']

w = f['block1_conv1']['block1_conv1_W_1:0']
b = f['block1_conv1']['block1_conv1_b_1:0']
model.layers[1].set_weights = [w,b]

w,b = f['block1_conv2']['block1_conv2_W_1:0'], f['block1_conv2']['block1_conv2_b_1:0']
model.layers[2].set_weights = [w,b]

w,b = f['block2_conv1']['block2_conv1_W_1:0'], f['block2_conv1']['block2_conv1_b_1:0']
model.layers[4].set_weights = [w,b]

w,b = f['block2_conv2']['block2_conv2_W_1:0'], f['block2_conv2']['block2_conv2_b_1:0']
model.layers[5].set_weights = [w,b]

f.close()
model.summary() 


from keras.optimizers import Adam

optimizer = Adam(lr = 0.0001, decay = 1e-5)


from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(patience = 5)


from keras.callbacks import ModelCheckpoint

check_point = ModelCheckpoint(filepath = 'best_model_todate', 
                             save_best_only = True, 
                             save_weights_only = True)

model.compile(loss = 'binary_crossentropy', metrics=['accuracy'], optimizer = optimizer)


# generate train data
batch_size = 20
epochs_size = 20

train_data_generator = data_generator(data = train_df, batch_size = batch_size)

train_steps = train_df.shape[0]//batch_size

print('Total training steps: ', train_steps)
print('Validation steps: ', len(valid_data))


# Fit the model
# =============================================================================
# fit_generator(generator, 
#                steps_per_epoch=None, 
#                epochs=1, 
#                verbose=1, 
#                callbacks=None, 
#                validation_data=None, 
#                validation_steps=None, 
#                validation_freq=1, 
#                class_weight=None, 
#                max_queue_size=10, 
#                workers=1, 
#                use_multiprocessing=False, 
#                shuffle=True, 
#                initial_epoch=0)
# 
# =============================================================================

# =============================================================================
# Trains the model on data generated batch-by-batch by a Python generator (or an instance of Sequence).
# 
# The generator is run in parallel to the model, for efficiency. For instance, this allows you to do real-time data augmentation on images on CPU in parallel to training your model on GPU.
# 
# The use of keras.utils.Sequence guarantees the ordering and guarantees the single use of every input per epoch when using use_multiprocessing=True.
# 
# Arguments
# 
#     generator: A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing. The output of the generator must be either
#         a tuple (inputs, targets)
#         a tuple (inputs, targets, sample_weights).
# 
#     This tuple (a single output of the generator) makes a single batch. Therefore, all arrays in this tuple must have the same length (equal to the size of this batch). Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size. The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
# 
#     steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to ceil(num_samples / batch_size) Optional for Sequence: if unspecified, will use the len(generator) as a number of steps.
#     epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire data provided, as defined by steps_per_epoch. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
#     verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
#     callbacks: List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks.
# 
#     validation_data: This can be either
#         a generator or a Sequence object for the validation data
#         tuple (x_val, y_val)
#         tuple (x_val, y_val, val_sample_weights)
# 
#     on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data.
# 
#     validation_steps: Only relevant if validation_data is a generator. Total number of steps (batches of samples) to yield from validation_data generator before stopping at the end of every epoch. It should typically be equal to the number of samples of your validation dataset divided by the batch size. Optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps.
#     validation_freq: Only relevant if validation data is provided. Integer or collections.Container instance (e.g. list, tuple, etc.). If an integer, specifies how many training epochs to run before a new validation run is performed, e.g. validation_freq=2 runs validation every 2 epochs. If a Container, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs validation at the end of the 1st, 2nd, and 10th epochs.
#     class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
#     max_queue_size: Integer. Maximum size for the generator queue. If unspecified, max_queue_size will default to 10.
#     workers: Integer. Maximum number of processes to spin up when using process-based threading. If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
#     use_multiprocessing: Boolean. If True, use process-based threading. If unspecified, use_multiprocessing will default to False. Note that because this implementation relies on multiprocessing, you should not pass non-picklable arguments to the generator as they can't be passed easily to children processes.
#     shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch. Only used with instances of Sequence (keras.utils.Sequence). Has no effect when steps_per_epoch is not None.
#     initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
# 
# Returns
# 
# A History object. Its History.history attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
# 
# Raises
# 
#     ValueError: In case the generator yields data in an invalid format.
# 
# Example
# 
# def generate_arrays_from_file(path):
#     while True:
#         with open(path) as f:
#             for line in f:
#                 # create numpy arrays of input data
#                 # and labels, from each line in the file
#                 x1, x2, y = process_line(line)
#                 yield ({'input_1': x1, 'input_2': x2}, {'output': y})
# 
# model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#                     steps_per_epoch=10000, epochs=10)
# 
# =============================================================================


history = model.fit_generator(train_data_generator, 
                              epochs = epochs_size, 
                              steps_per_epoch = train_steps, 
                              validation_data = (valid_data, valid_labels), 
                              callbacks = [early_stopping, check_point], 
                              class_weight={0:1.0, 1:0.4})

'''
    Why callbacks? Early Stopping? Checkpoints? Etc?
    What is class_weight doing here?
    How class_weight works?
'''


# Preparing test data
normal_dir = test_dir / 'NORMAL'
pneumonia_dir = test_dir / 'PNEUMONIA'

print(normal_dir)
print(pneumonia_dir)


normal_img_obj = normal_dir.glob('*.jpeg')
pneumonia_img_obj = pneumonia_dir.glob('*.jpeg')


print(normal_img_obj)
print(pneumonia_img_obj)


test_data = []
test_labels = []
    
for img in normal_img_obj:
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert to float 32 and scale
    img = img.astype(np.float32)/255.0
    
    # Categorize label
    label = to_categorical(0, num_classes = 2)
    
    test_data.append(img)
    test_labels.append(label)
    
for img in pneumonia_img_obj:
    img = cv.imread(str(img))
    img = cv.resize(img, (128, 128))
    
    if img.shape[2] == 1:
        img = np.dstack[img, img, img]
    else: 
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
    
    img = img.astype(np.float32)/255.0
    
    
    label = to_categorical(1, num_classes = 2)
    
    test_data.append(img)    
    test_labels.append(label)
    

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print('Test data shape: ', test_data.shape)
print('Test labels shape: ', test_labels.shape)


# Evaluate model
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size = 12)
print('Test loss', test_loss)
print('Test score', test_score)

from keras.models import Model
from keras.models import load_model



# keras faq
# https://keras.io/getting-started/faq/
del model
from keras.models import load_model

pretrained_model_path = '..\\chest_xray\\best_model.hdf5'
print(pretrained_model_path)

pretrained_model = load_model(pretrained_model_path)
pretrained_model_weights = pretrained_model.load_weights(pretrained_model_path)


test_loss, test_score = pretrained_model.evaluate(test_data, test_labels, batch_size = 12)
print('Test loss', test_loss)
print('Test score', test_score)



# Get predictions
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(test_labels, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)


# Get the confusion matrix
cm  = confusion_matrix(orig_test_labels, preds)
plt.figure()
plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True, alpha=0.7,cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.show()


# Calculate Precision and Recall
tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))
for size in [2, 5, 8]:
    print(size)
    
for _ in range(8):
    print('a')


'''
Here's my birthday gift analogy: You have a list of things you want for your birthday. If your friends get you all of those things, that's 100% recall, even if they get you some other presents as well. If instead they just bought you other stuff you didn't have on your list (or no presents at all) that is 0% recall.

Precision is about only getting the gifts you want. If you only get one present, but it's on your list, that's 100% precision. Same if every single gift you receive is on your list. Precision starts going down when they buy you things not on your list. If half the gifts you get aren't on your list, that's 50% precision. If none of the gifts you get are on your list, then that's 0% precision.
'''

'''


I found this explanation by one Nathan Yan on Quora

Top-N accuracy means that the correct class gets to be in the Top-N probabilities for it to count as “correct”. As an example, suppose I have a data set of images

    Dog
    Cat
    Dog
    Bird
    Cat
    Cat
    Mouse
    Penguin

For each of these, the model will predict a class, which will appear next to the correct class in quotations

    Dog “Dog”
    Cat “Bird”
    Dog “Dog”
    Bird “Bird”
    Cat “Cat”
    Cat “Cat”
    Mouse “Penguin”
    Penguin “Dog”

The Top-1 accuracy for this is (5 correct out of 8), 62.5%. Now suppose I also list out the rest of the classes the model predicted, in descending order of their probabilities (the further right the class appears, the less likely the model thinks the image is tat class)

- Dog “Dog, Cat, Bird, Mouse, Penguin”
- Cat “Bird, Mouse, Cat, Penguin, Dog”
- Dog “Dog, Cat, Bird, Penguin, Mouse”
- Bird “Bird, Cat, Mouse, Penguin, Dog”
- Cat “Cat, Bird, Mouse, Dog, Penguin”
- Cat “Cat, Mouse, Dog, Penguin, Bird”
- Mouse “Penguin, Mouse, Cat, Dog, Bird”
- Penguin “Dog, Mouse, Penguin, Cat, Bird”

If we take the top-3 accuracy for this, the correct class only needs to be in the top three predicted classes to count. As a result, despite the model not perfectly getting every problem, its top-3 accuracy is 100%!



'''
'''


Top-1 accuracy is the conventional accuracy: the model answer (the one with highest probability) must be exactly the expected answer.

Top-5 accuracy means that any of your model 5 highest probability answers must match the expected answer.

For instance, let's say you're applying machine learning to object recognition using a neural network. A picture of a cat is shown, and these are the outputs of your neural network:

    Tiger: 0.4
    Dog: 0.3
    Cat: 0.1
    Lynx: 0.09
    Lion: 0.08
    Bird: 0.02
    Bear: 0.01

Using top-1 accuracy, you count this output as wrong, because it predicted a tiger.

Using top-5 accuracy, you count this output as correct, because cat is among the top-5 guesses.

'''
'''
Is accuracy = 1- test error rate?


In principle yes, accuracy is the fraction of properly predicted cases thus 1-the fraction of misclassified cases, that is error (rate). Both terms may be sometimes used in a more vague way, however, and cover different things like class-balanced error/accuracy or even F-score or AUROC -- it is always best to look for/include a proper clarification in the paper or report.

Also note that test error rate implies error on a test set, so it is likely 1-test set accuracy, and there may be other accuracies flying around.

'''

'''
Data augmentation

We augmented the data to artificially increase the size of the dataset. We used various affine transforms, and gradually increased the intensity of the augmentation as our models started to overfit more. We ended up with some pretty extreme augmentation parameters:

    rotation: random with angle between 0° and 360° (uniform)
    translation: random with shift between -10 and 10 pixels (uniform)
    rescaling: random with scale factor between 1/1.6 and 1.6 (log-uniform)
    flipping: yes or no (bernoulli)
    shearing: random with angle between -20° and 20° (uniform)
    stretching: random with stretch factor between 1/1.3 and 1.3 (log-uniform)

'''
''' Adam vs SGD 
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

Adam [1] is an adaptive learning rate optimization algorithm that’s been designed 
specifically for training deep neural networks. First published in 2014, Adam was 
presented at a very prestigious conference for deep learning practitioners — ICLR 2015. 
The paper contained some very promising diagrams, showing huge performance gains 
in terms of speed of training. However, after a while people started noticing, 
that in some cases Adam actually finds worse solution than stochastic gradient 
descent. A lot of research has been done to address the problems of Adam.

Nitish Shirish Keskar and Richard Socher in their paper ‘Improving Generalization 
Performance by Switching from Adam to SGD’ [5] also showed that by switching to 
SGD during training training they’ve been able to obtain better generalization 
power than when using Adam alone. They proposed a simple fix which uses a very 
simple idea. They’ve noticed that in earlier stages of training Adam still 
outperforms SGD but later the learning saturates. They proposed simple strategy 
which they called SWATS in which they start training deep neural network with Adam 
but then switch to SGD when certain criteria hits. They managed to achieve results 
comparable to SGD with momentum.
'''

''' loss function
What’s a Loss Function?

At its core, a loss function is incredibly simple: it’s a method of evaluating how well your algorithm models your dataset. If your predictions are totally off, your loss function will output a higher number. If they’re pretty good, it’ll output a lower number. As you change pieces of your algorithm to try and improve your model, your loss function will tell you if you’re getting anywhere.

'''
''' optimizer 
During the training process, we tweak and change the parameters (weights) of our model to try and minimize that loss function, and make our predictions as correct as possible. But how exactly do you do that? How do you change the parameters of your model, by how much, and when?

This is where optimizers come in. They tie together the loss function and model parameters by updating the model in response to the output of the loss function. In simpler terms, optimizers shape and mold your model into its most accurate possible form by futzing with the weights. The loss function is the guide to the terrain, telling the optimizer when it’s moving in the right or wrong direction.

For a useful mental model, you can think of a hiker trying to get down a mountain with a blindfold on. It’s impossible to know which direction to go in, but there’s one thing she can know: if she’s going down (making progress) or going up (losing progress). Eventually, if she keeps taking steps that lead her downwards, she’ll reach the base.

Similarly, it’s impossible to know what your model’s weights should be right from the start. But with some trial and error based on the loss function (whether the hiker is descending), you can end up getting there eventually.

'''

''' Batch Normalization 
An easy way to solve this problem for the input layer is to randomize the data 
before creating mini-batches.

But, how do we solve this for the hidden layers? Just as it made intuitive sense 
to have a uniform distribution for the input layer, it is advantageous to have 
the same input distribution for each hidden unit over time while training. 
But in a neural network, each hidden unit’s input distribution changes every time 
there is a parameter update in the previous layer. 

This is called internal covariate shift. This makes training slow and requires 
a very small learning rate and a good parameter initialization. 

This problem is solved by normalizing the layer’s inputs over a mini-batch and 
this process is therefore called Batch Normalization.

'''


'''
Class SeparableConv2D
Aliases:

    Class tf.keras.layers.SeparableConv2D
    Class tf.keras.layers.SeparableConvolution2D

Defined in tensorflow/python/keras/layers/convolutional.py.

Depthwise separable 2D convolution.

Separable convolutions consist in first performing a depthwise spatial convolution (which acts on each input channel separately) followed by a pointwise convolution which mixes together the resulting output channels. The depth_multiplier argument controls how many output channels are generated per input channel in the depthwise step.

Intuitively, separable convolutions can be understood as a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
Arguments:

    filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
    strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    padding: one of "valid" or "same" (case-insensitive).
    data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".
    dilation_rate: An integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any strides value != 1.
    depth_multiplier: The number of depthwise convolution output channels for each input channel. The total number of depthwise convolution output channels will be equal to filters_in * depth_multiplier.
    activation: Activation function to use. If you don't specify anything, no activation is applied (ie. "linear" activation: a(x) = x).
    use_bias: Boolean, whether the layer uses a bias vector.
    depthwise_initializer: Initializer for the depthwise kernel matrix.
    pointwise_initializer: Initializer for the pointwise kernel matrix.
    bias_initializer: Initializer for the bias vector.
    depthwise_regularizer: Regularizer function applied to the depthwise kernel matrix.
    pointwise_regularizer: Regularizer function applied to the pointwise kernel matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
    depthwise_constraint: Constraint function applied to the depthwise kernel matrix.
    pointwise_constraint: Constraint function applied to the pointwise kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.

Input shape: 4D tensor with shape: (batch, channels, rows, cols) if data_format='channels_first' or 4D tensor with shape: (batch, rows, cols, channels) if data_format='channels_last'.

Output shape: 4D tensor with shape: (batch, filters, new_rows, new_cols) if data_format='channels_first' or 4D tensor with shape: (batch, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.
__init__

__init__(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    depth_multiplier=1,
    activation=None,
    use_bias=True,
    depthwise_initializer='glorot_uniform',
    pointwise_initializer='glorot_uniform',
    bias_initializer='zeros',
    depthwise_regularizer=None,
    pointwise_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    depthwise_constraint=None,
    pointwise_constraint=None,
    bias_constraint=None,
    **kwargs
)

'''
'''
https://arxiv.org/abs/1610.02357
Convolutional neural networks have emerged as the master
algorithm in computer vision in recent years, and developing
recipes for designing them has been a subject of
considerable attention. The history of convolutional neural
network design started with LeNet-style models [10], which
were simple stacks of convolutions for feature extraction
and max-pooling operations for spatial sub-sampling. In
2012, these ideas were refined into the AlexNet architecture
[9], where convolution operations were being repeated
multiple times in-between max-pooling operations, allowing
the network to learn richer features at every spatial scale.
What followed was a trend to make this style of network
increasingly deeper, mostly driven by the yearly ILSVRC
competition; first with Zeiler and Fergus in 2013 [25] and
then with the VGG architecture in 2014 [18].
At this point a new style of network emerged, the Inception
architecture, introduced by Szegedy et al. in 2014 [20]
as GoogLeNet (Inception V1), later refined as Inception V2
[7], Inception V3 [21], and most recently Inception-ResNet
[19]. Inception itself was inspired by the earlier Network-
In-Network architecture [11]. Since its first introduction,
Inception has been one of the best performing family of
models on the ImageNet dataset [14], as well as internal
datasets in use at Google, in particular JFT [5].
The fundamental building block of Inception-style models
is the Inception module, of which several different versions
exist. In figure 1 we show the canonical form of an
Inception module, as found in the Inception V3 architecture.
An Inception model can be understood as a stack of
such modules. This is a departure from earlier VGG-style
networks which were stacks of simple convolution layers.
While Inception modules are conceptually similar to convolutions
(they are convolutional feature extractors), they
empirically appear to be capable of learning richer representations
with less parameters. How do they work, and
how do they differ from regular convolutions? What design
strategies come after Inception?


A complete description of the specifications of the network
is given in figure 5. The Xception architecture has
36 convolutional layers forming the feature extraction base
of the network. In our experimental evaluation we will exclusively
investigate image classification and therefore our
convolutional base will be followed by a logistic regression
layer. Optionally one may insert fully-connected layers before
the logistic regression layer, which is explored in the
experimental evaluation section (in particular, see figures
7 and 8). The 36 convolutional layers are structured into
14 modules, all of which have linear residual connections
around them, except for the first and last modules.

4.1. The JFT dataset
JFT is an internal Google dataset for large-scale image
classification dataset, first introduced by Hinton et al. in [5],
which comprises over 350 million high-resolution images
annotated with labels from a set of 17,000 classes. To evaluate
the performance of a model trained on JFT, we use an
auxiliary dataset, FastEval14k.
FastEval14k is a dataset of 14,000 images with dense
annotations from about 6,000 classes (36.5 labels per image
on average). On this dataset we evaluate performance
using Mean Average Precision for top 100 predictions
(MAP@100), and we weight the contribution of each class
to MAP@100 with a score estimating how common (and
therefore important) the class is among social media images.
This evaluation procedure is meant to capture performance
on frequently occurring labels from social media, which is
crucial for production models at Google.

4.2. Optimization configuration
A different optimization configuration was used for ImageNet
and JFT:
 On ImageNet:
– Optimizer: SGD
– Momentum: 0.9
– Initial learning rate: 0.045
– Learning rate decay: decay of rate 0.94 every 2
epochs
 On JFT:
– Optimizer: RMSprop [22]
– Momentum: 0.9
– Initial learning rate: 0.001
– Learning rate decay: decay of rate 0.9 every
- 3,000,000 samples


4.3. Regularization configuration
 Weight decay: The Inception V3 model uses a weight
decay (L2 regularization) rate of 4e 􀀀 5, which has
been carefully tuned for performance on ImageNet. We
found this rate to be quite suboptimal for Xception
and instead settled for 1e 􀀀 5. We did not perform
an extensive search for the optimal weight decay rate.
The same weight decay rates were used both for the
ImageNet experiments and the JFT experiments.
 Dropout: For the ImageNet experiments, both models
include a dropout layer of rate 0.5 before the logistic
regression layer. For the JFT experiments, no dropout
was included due to the large size of the dataset which
made overfitting unlikely in any reasonable amount of
time.
 Auxiliary loss tower: The Inception V3 architecture
may optionally include an auxiliary tower which backpropagates
the classification loss earlier in the network,
serving as an additional regularization mechanism. For
simplicity, we choose not to include this auxiliary tower
in any of our models.


4.4. Training infrastructure
All networks were implemented using the TensorFlow
framework [1] and trained on 60 NVIDIA K80 GPUs each.
For the ImageNet experiments, we used data parallelism
with synchronous gradient descent to achieve the best classification
performance, while for JFT we used asynchronous
gradient descent so as to speed up training. The ImageNet
experiments took approximately 3 days each, while the JFT
experiments took over one month each. The JFT models
were not trained to full convergence, which would have
taken over three month per experiment.

Top-1 accuracy Top-5 accuracy
VGG-16 0.715 0.901
ResNet-152 0.770 0.933
Inception V3 0.782 0.941
Xception 0.790 0.945



Documentation for individual models
Model 	Size 	Top-1 Accuracy 	Top-5 Accuracy 	Parameters 	Depth
Xception 	88 MB 	0.790 	0.945 	22,910,480 	126
VGG16 	528 MB 	0.713 	0.901 	138,357,544 	23
VGG19 	549 MB 	0.713 	0.900 	143,667,240 	26
ResNet50 	98 MB 	0.749 	0.921 	25,636,712 	-
ResNet101 	171 MB 	0.764 	0.928 	44,707,176 	-
ResNet152 	232 MB 	0.766 	0.931 	60,419,944 	-
ResNet50V2 	98 MB 	0.760 	0.930 	25,613,800 	-
ResNet101V2 	171 MB 	0.772 	0.938 	44,675,560 	-
ResNet152V2 	232 MB 	0.780 	0.942 	60,380,648 	-
ResNeXt50 	96 MB 	0.777 	0.938 	25,097,128 	-
ResNeXt101 	170 MB 	0.787 	0.943 	44,315,560 	-
InceptionV3 	92 MB 	0.779 	0.937 	23,851,784 	159
InceptionResNetV2 	215 MB 	0.803 	0.953 	55,873,736 	572
MobileNet 	16 MB 	0.704 	0.895 	4,253,864 	88
MobileNetV2 	14 MB 	0.713 	0.901 	3,538,984 	88
DenseNet121 	33 MB 	0.750 	0.923 	8,062,504 	121
DenseNet169 	57 MB 	0.762 	0.932 	14,307,880 	169
DenseNet201 	80 MB 	0.773 	0.936 	20,242,984 	201
NASNetMobile 	23 MB 	0.744 	0.919 	5,326,716 	-
NASNetLarge 	343 MB 	0.825 	0.960 	88,949,818 	-

'''            
            
'''    



It's easier to understand what np.vstack, np.hstack and np.dstack* do by looking at the .shape attribute of the output array.

Using your two example arrays:

print(a.shape, b.shape)
# (3, 2) (3, 2)

    np.vstack concatenates along the first dimension...

    print(np.vstack((a, b)).shape)
    # (6, 2)

    np.hstack concatenates along the second dimension...

    print(np.hstack((a, b)).shape)
    # (3, 4)

    and np.dstack concatenates along the third dimension.

    print(np.dstack((a, b)).shape)
    # (3, 2, 2)

Since a and b are both two dimensional, np.dstack expands them by inserting a third dimension of size 1. This is equivalent to indexing them in the third dimension with np.newaxis (or alternatively, None) like this:

print(a[:, :, np.newaxis].shape)
# (3, 2, 1)

If c = np.dstack((a, b)), then c[:, :, 0] == a and c[:, :, 1] == b.

You could do the same operation more explicitly using np.concatenate like this:

print(np.concatenate((a[..., None], b[..., None]), axis=2).shape)
# (3, 2, 2)

* Importing the entire contents of a module into your global namespace using import * is considered bad practice for several reasons. The idiomatic way is to import numpy as np.

'''    
data = []
labels = []

# Read all images inside Parasitized Path
for uninfected in uninfected_path:
    try:
        image = cv.imread('..\\cell_images\\Uninfected\\' + uninfected)        
        
        # Convert to PIL array 
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize image
        img_resized_np = img_pil_array.resize((64, 64))
        
        # append to data
        data.append(np.array(img_resized_np))
        
        # Label the image as 1 = uninfected
        labels.append(0)
        
    except AttributeError:
        print('Error exception uninfected_path')

for parasitized in parasitized_path:
    try:
        image = cv.imread('..\\cell_images\\Parasitized\\'+ parasitized)
        # Convert to numpy array
        img_pil_array = Image.fromarray(image, 'RGB')
        
        # Resize all images to get same size
        img_resized_np = img_pil_array.resize((64, 64))
        
        # alternative approach: using sklearn to resize   
        #from skimage.transform import resize
        #img_resized_sklearn = resize(image, (64, 64), anti_aliasing=True)
    
        # or you can resize image using openCV
        # you need to convert it to an array then, you can append to data array
        #img_resized_opncv = cv.resize(image, dsize=(64, 64), interpolation=cv.INTER_CUBIC)
        
        # append all images into single array    
        data.append(np.array(img_resized_np))
        
        '''
        How can we track parasitized and normal?
        We are using all parasitized as label 1
        and all uninfected as label 1
        So, if the label is 1, it is parasitized
        '''
        labels.append(1) 
    except AttributeError:
        print('Error exeption parasitized_path')
    
#print(data[:2])
'''
To do    
1. Use openCV to resize image
2. Check PIL vs OpenCV for resize and array conversion

'''
print(data[1]) # numpy array
print(labels[1]) # 0

print(len(data)) # 27558
print(len(labels)) # 27558


#Shape of the data
# data = np.array(data)
# labels = np.array(labels)

print("Shape of the data array: ", np.shape(data))
print("Shape of the label array: ", np.shape(labels))


# Save image array to use later. Made it easy
cells = np.array(data)
labels = np.array(labels)

np.save('Cells' , cells)
np.save('Labels' , labels)


print('Cells : {} | labels : {}'.format(cells.shape , labels.shape))

print(cells.shape) # (27558, 64, 64, 3)
print(cells.shape[0]) # 27558

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    
    # Take random image
    r = np.random.randint(0 , cells.shape[0] , 1)
    
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    
    plt.imshow(cells[r[0]])
    
    plt.title('{} : {}'
              .format('Infected' if labels[r[0]] == 1 
                                  else 'Unifected' , labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()

plt.figure(1, figsize = (15, 9))
n = 0
for i in range(16):
    
    n += 1
    
    # Take a random number in each iteration
    # np.random.randint(low, high, dtType)
    r = np.random.randint(0, cells.shape[0], 1)
    
    # Create subplot
    plt.subplot(4, 4, n)
    
    # Adjust subplots
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
    
    # Show single image using random selection
    # For each iteration, random number will be selected
    # image will be shown according to random numbered
    plt.imshow(cells[ r[0] ])
    
    
    # Show title
    plt.title('{} : {}'.format(
            'Infected' if labels[r[0]] == 1
            else 'Uninfected', labels[r[0]]
            ))
    
    plt.xticks([]), plt.yticks([])
plt.show()


print(cells.shape) # 27558, 64, 64, 3
plt.figure(1, figsize = (10 , 7))
plt.subplot(1 , 2 , 1)
plt.imshow(cells[0])
plt.title('{} : {}'.format(
            'Infected' if labels[0] == 1
            else 'Uninfected', labels[0]
            ))
plt.xticks([]) , plt.yticks([])

plt.subplot(1 , 2 , 2)
plt.imshow(cells[26356])
plt.title('{} : {}'.format(
            'Infected' if labels[26356] == 1
            else 'Uninfected', labels[26356]
            ))
plt.xticks([]) , plt.yticks([])
plt.show()



# Load from the saved numpy array cells and labels
cells_loaded=np.load("Cells.npy")
labels_loaded=np.load("Labels.npy")

print(cells_loaded.shape[0]) # 27558

# Arrange 
shuffled_cells = np.arange(cells_loaded.shape[0])
print(shuffled_cells)

# Random shuffle
np.random.shuffle(shuffled_cells)
print(shuffled_cells)


print(cells_loaded[shuffled_cells])
print(labels_loaded[shuffled_cells])

cells_randomly_shuffled = cells_loaded[shuffled_cells]
labels_randomly_shuffled = labels_loaded[shuffled_cells]

print(np.unique(labels_randomly_shuffled)) # [0 1]
print(len(cells_randomly_shuffled)) # 27558

num_classes = len(np.unique(labels_randomly_shuffled))
print(num_classes)

len_data = len(cells_randomly_shuffled)
print(len_data)

print(0.1*len_data) # 2755.8

print(len(cells_randomly_shuffled[(int)(0.1*len_data):])) # 24803
print(len(cells_randomly_shuffled[:(int)(0.1*len_data)])) # 2755


''' Train Test Split Technique 1'''
(x_train, x_test) = cells_randomly_shuffled[(int)(0.1*len_data):], cells_randomly_shuffled[:(int)(0.1*len_data)]
print(len(x_train))
print(len(x_test))

# As we are working on image data we are normalizing data by divinding 255.
x_train = x_train.astype('float32')/255 
print(x_train)

x_test = x_test.astype('float32')/255

x_train_len = len(x_train)
x_test_len = len(x_test)
print(x_train_len)
print(x_test_len)

(y_train, y_test) = labels_randomly_shuffled[(int)(0.1*len_data):], labels_randomly_shuffled[:(int)(0.1*len_data)]

''' sklearn train test split version
from sklearn.model_selection import train_test_split

train_x , x , train_y , y = train_test_split(cells , labels , 
                                            test_size = 0.2 ,
                                            random_state = 111)

eval_x , test_x , eval_y , test_y = train_test_split(x , y , 
                                                    test_size = 0.5 , 
                                                    random_state = 111)

How to split .2, .2, 6?
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test 
= train_test_split(X, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val 
= train_test_split(X_train, y_train, test_size=0.25, random_state=1)

'''

#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

print(y_train)
print(len(y_train))
print(y_test)
print(len(y_test))



# Create model
model = Sequential()

'''
api reference r1.13
tf.layers.conv2d(
    inputs,
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format='channels_last',
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
    
    
)

What does conv2D do in tensorflow?
https://stackoverflow.com/questions/34619177/what-does-tf-nn-conv2d-do-in-tensorflow

'''
# Add hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(64,64,3)))

# Step 2 - Max Pooling - Taking the maximum
# Why? Reduce the number of nodes for next Flattening step
model.add(MaxPooling2D(pool_size = (2, 2)))

# Add another hidden layer
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))


model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# Droput to avoid overfitting
model.add(Dropout(0.2))

# Step 3 - Flatten - huge single 1 dimensional vector
model.add(Flatten())

# Step 4 - Full Connection
# output_dim/units: don't take too small, don't take too big
# common practice take a power of 2, such as 128, 256, etc.
model.add(Dense(units = 128, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax")) 

model.summary()


# Step 5
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Step 6 Fitting model
model.fit(x_train, y_train, batch_size=50, epochs=20, verbose=1)

# Check accuracy
accuracy = model.evaluate(x_test, y_test, verbose=1)

# Save model weights
from keras.models import load_model
model.save('malaria_tfcnnsoftmax_category.h5')

# Use model using tkinter
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import cv2 as cv

def convert_to_array(img):
    img = cv.imread(img)
    img = Image.fromarray(img)
    img = img.resize(64, 64)
    return np.array(img)

def get_label(label):
    if label == 0:
        return 'Uninfected'
    if label == 1:
        return 'Parasitized'

def predict_malaria(img_file):
    
    model = load_model('malaria_tfcnnsoftmax_category.h5')
    
    print('Predciting Malaria....')
    
    img_array = convert_to_array(img_file)
    img_array = img_array/255
    
    img_data = []
    img_data.append(img_array)
    img_data = np.array(img_data)
    
    score = model.predict(img_data, verbose=1)
    print('Score', score)
    
    label_index = np.argmax(score)
    
    result = get_label(label_index)
    return result, 'Predicted image is : ' + result + 'with accuracy = ' + str(accuracy)


"""from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter import filedialog 
from tkinter import messagebox as mbox

class Example(Frame):

    def __init__(self):
        super().__init__()   

        self.initUI()


    def initUI(self):

        self.master.title("File dialog")
        self.pack(fill=BOTH, expand=1)

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)        

        

    def onOpen(self):

        ftypes = [('Image', '*.png'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes = ftypes)
        fl = dlg.show()
        c,s=predict_cell(fl)
        root = Tk()
        T = Text(root, height=4, width=70)
        T.pack()
        T.insert(END, s)
        

def main():

    root = Tk()
    ex = Example()
    root.geometry("100x50+100+100")
    root.mainloop()  


if __name__ == '__main__':
    main()"""
    
'''
Note:
Kernel size vs filter?

https://stackoverflow.com/questions/51180234/keras-conv2d-filters-vs-kernel-size


Each convolution layer consists of several convolution channels (aka. depth or filters). 
In practice, they are in number of 64,128,256, 512 etc. This is equal to 
number of channels in output of convolution layer. kernel_size on the other 
hand is size of these convolution filters. In practice they are 3x3, 1x1 or 5x5. 
As abbreviation, they could be written as 1 or 3 or 5 as they are mostly square 
in practice.

Edit

Following quote should make it more clear.

Discussion on vlfeat

Suppose X is an input with size W x H x D x N (where N is the size of the batch) 
to a convolutional layer containing filters F (with size FW x FH x FD x K) in a network.

The number of feature channels D is the third dimension of the input X here 
(for example, this is typically 3 at the first input to the network if the 
input consists of colour images). The number of filters K is the fourth dimension 
of F. The two concepts are closely linked because if the number of filters in a 
layer is K, it produces an output with K feature channels. So the input to the 
next layer will have K feature channels.

The FW x FH above is filter size you are looking for.

Added

You should be familiar with filters. You can consider each filter to be responsible
ible to extract some type of feature from raw image. The CNNs try to learn such 
filters i.e. the filters are parametrized in CNNs are learned during training of 
CNNs. These are filters in CNN. You apply each filter in a Conv2D to each input 
channel and combine these to get output channels. So, number of filter and output 
channels are same.
    
'''
    