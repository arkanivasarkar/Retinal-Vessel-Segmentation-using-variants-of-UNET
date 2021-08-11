import os
import cv2
import numpy as np
import skimage.io
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
np.random.seed(0)

# CLAHE
def clahe_equalized(imgs):    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))    
    imgs_equalized = clahe.apply(imgs)
    return imgs_equalized

patch_size = 512

#loading model architectures
from model import unetmodel, residualunet, attentionunet, attention_residualunet
from tensorflow.keras.optimizers import Adam
from evaluation_metrics import IoU_coef,IoU_loss

IMG_HEIGHT = patch_size
IMG_WIDTH = patch_size
IMG_CHANNELS = 1

input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = unetmodel(input_shape) #/residualunet(input_shape)/attentionunet(input_shape)/attention_residualunet(input_shape) 
model.compile(optimizer = Adam(lr = 1e-3), loss= IoU_loss, metrics= ['accuracy', IoU_coef])
model.load_weights('/content/drive/MyDrive/training/retina_Unet_150epochs.hdf5') #loading weights


path1 = '/content/drive/MyDrive/training/images'    #test dataset images directory path
path2 = '/content/drive/MyDrive/training/masks'     #test dataset mask directory path


from sklearn.metrics import jaccard_score,confusion_matrix

testimg = []
ground_truth = []
prediction = []
global_IoU = []
global_accuracy = []

testimages = sorted(os.listdir(path1))
testmasks =  sorted(os.listdir(path2))

for idx, image_name in enumerate(testimages):  
   if image_name.endswith(".jpg"):  
      predicted_patches = []
      test_img = skimage.io.imread(path1+"/"+image_name)
     
      test = test_img[:,:,1] #selecting green channel
      test = clahe_equalized(test) #applying CLAHE
      SIZE_X = (test_img.shape[1]//patch_size)*patch_size #getting size multiple of patch size
      SIZE_Y = (test_img.shape[0]//patch_size)*patch_size #getting size multiple of patch size
      test = cv2.resize(test, (SIZE_X, SIZE_Y))
      testimg.append(test)           
      test = np.array(test)

      patches = patchify(test, (patch_size, patch_size), step=patch_size) #create patches(patch_sizexpatch_sizex1)

      for i in range(patches.shape[0]):
                for j in range(patches.shape[1]):
                  single_patch = patches[i,j,:,:]
                  single_patch_norm = (single_patch.astype('float32')) / 255.
                  single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
                  single_patch_input = np.expand_dims(single_patch_norm, 0)
                  single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) #predict on single patch
                  predicted_patches.append(single_patch_prediction)
      predicted_patches = np.array(predicted_patches)
      predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
      reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape) #join patches to form whole img
      prediction.append(reconstructed_image) 

      groundtruth=[]
      groundtruth = skimage.io.imread(path2+'/'+testmasks[idx]) #reading mask of the test img      
      SIZE_X = (groundtruth.shape[1]//patch_size)*patch_size 
      SIZE_Y = (groundtruth.shape[0]//patch_size)*patch_size  
      groundtruth = cv2.resize(groundtruth, (SIZE_X, SIZE_Y))  
      ground_truth.append(groundtruth)

      y_true = groundtruth
      y_pred = reconstructed_image
      labels = [0, 1]
      IoU = []
      for label in labels:
          jaccard = jaccard_score(y_pred.flatten(),y_true.flatten(), pos_label=label, average='weighted')
          IoU.append(jaccard)
      IoU = np.mean(IoU) #jacard/IoU of single image
      global_IoU.append(IoU)

      cm=[]
      accuracy = []
      cm = confusion_matrix(y_true.flatten(),y_pred.flatten(), labels=[0, 1])
      accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1]) #accuracy of single image
      global_accuracy.append(accuracy)


avg_acc =  np.mean(global_accuracy)
mean_IoU = np.mean(global_IoU)

print('Average accuracy is',avg_acc)
print('mean IoU is',mean_IoU)


#checking segmentation results
import random
test_img_number = random.randint(0, len(testimg))
plt.figure(figsize=(20, 18))
plt.subplot(231)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.imshow(testimg[test_img_number])
plt.subplot(232)
plt.title('Ground Truth')
plt.xticks([])
plt.yticks([])
plt.imshow(ground_truth[test_img_number],cmap='gray')
plt.subplot(233)
plt.title('Prediction')
plt.xticks([])
plt.yticks([])
plt.imshow(prediction[test_img_number],cmap='gray')

plt.show()



#prediction on single image
from datetime import datetime 
reconstructed_image = []
test_img = skimage.io.imread('/content/drive/MyDrive/hrf/images/15_dr.jpg') #test image

predicted_patches = []
start = datetime.now()   

test = test_img[:,:,1] #selecting green channel
test = clahe_equalized(test) #applying CLAHE
SIZE_X = (test_img.shape[1]//patch_size)*patch_size #getting size multiple of patch size
SIZE_Y = (test_img.shape[0]//patch_size)*patch_size #getting size multiple of patch size
test = cv2.resize(test, (SIZE_X, SIZE_Y))        
test = np.array(test)
patches = patchify(test, (patch_size, patch_size), step=patch_size) #create patches(patch_sizexpatch_sizex1)

for i in range(patches.shape[0]):
      for j in range(patches.shape[1]):
          single_patch = patches[i,j,:,:]
          single_patch_norm = (single_patch.astype('float32')) / 255.
          single_patch_norm = np.expand_dims(np.array(single_patch_norm), axis=-1)
          single_patch_input = np.expand_dims(single_patch_norm, 0)
          single_patch_prediction = (model.predict(single_patch_input)[0,:,:,0] > 0.5).astype(np.uint8) #predict on single patch
          predicted_patches.append(single_patch_prediction)
predicted_patches = np.array(predicted_patches)
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], patch_size,patch_size) )
reconstructed_image = unpatchify(predicted_patches_reshaped, test.shape) #join patches to form whole img

stop = datetime.now()
print('Execution time: ',(stop-start)) #computation time

plt.subplot(121)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])
plt.imshow(test_img)
plt.subplot(122)
plt.title('Prediction')
plt.xticks([])
plt.yticks([])
plt.imshow(reconstructed_image,cmap='gray')

plt.show()
