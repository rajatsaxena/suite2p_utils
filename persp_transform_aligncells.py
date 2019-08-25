#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.measure import ransac
from skimage.feature import (plot_matches, match_descriptors, ORB)
from skimage.transform import warp
from scipy.linalg import norm
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


def transformImages(img1,img2,transformation='projective'):
    # laod the ORB descriptor
    descriptor_extractor = ORB(n_keypoints=500)

    # find keypoints for both the images
    descriptor_extractor.detect_and_extract(img1)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img2)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # find matches bw the keypoints from both the images
    matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
    
    # find matching coordinates between image1 and image2
    matches_img1 = keypoints1[matches12[:,0]]
    matches_img2 = keypoints2[matches12[:,1]]

    # estimate affine transform model using matching coordinates between image1 and image2
#    model = AffineTransform()
#    model.estimate(matches_img1, matches_img2)

    # robustly estimate affine transform model with RANSAC
    if transformation=='affine':
        model_robust, inliers = ransac((matches_img1, matches_img2), AffineTransform, min_samples=4,
                                   residual_threshold=6, max_trials=100)
    else:
        model_robust, inliers = ransac((matches_img1, matches_img2), ProjectiveTransform, min_samples=4,
                                   residual_threshold=6, max_trials=100)
    outliers = inliers == False
    
    print("Num of inliers: ", str(len(np.where(inliers)[0])))
    
    # visualize correspondence after ransac algorithm
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4,8))
    plt.gray()
    plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
    ax[0].set_title('All correspondences')
    ax[0].axis('off')
    inlier_idxs = np.nonzero(inliers)[0]
    plot_matches(ax[1], meanImg1, meanImg2, matches_img1, matches_img2, np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
    ax[1].set_title('Correct correspondences')
    ax[1].axis('off')
    outlier_idxs = np.nonzero(outliers)[0]
    plot_matches(ax[2], meanImg1, meanImg2, matches_img1, matches_img2, np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
    ax[2].set_title('Faulty correspondences')
    ax[2].axis('off')
    plt.tight_layout()
    plt.show()
    
    # warp image1 wrt image2
    meanImg2_transformed = warp(meanImg2, model_robust.inverse, cval=-1)
    
    return meanImg2_transformed, model_robust


# In[5]:


# load the data and mean images
ops1 = np.load('ops1.npy', allow_pickle=True).item()
ops2 = np.load('ops2.npy', allow_pickle=True).item()

meanImg1 = ops1['meanImg']
meanImg2 = ops2['meanImg']


# In[25]:


# plot the raw data
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.imshow(meanImg1, cmap='gray')
plt.title("Session1 Mean Image", fontsize=12)
plt.axis('off')
plt.subplot(122)
plt.imshow(meanImg2, cmap='gray')
plt.title("Session2 Mean Image", fontsize=12)
plt.axis('off')
plt.tight_layout()


# In[32]:


meanImg2_t, model_proj = transformImages(meanImg1,meanImg2,'projective')


# In[44]:


# plot the processed data
plt.figure(figsize=(12,6))
plt.subplot(231)
plt.imshow(meanImg1, cmap='gray')
plt.title("Mean Image1", fontsize=12)
plt.axis('off')
plt.subplot(232)
plt.imshow(meanImg2, cmap='gray')
plt.title("Mean Image2", fontsize=12)
plt.axis('off')
plt.subplot(233)
plt.imshow(meanImg2_t, cmap='gray')
plt.title("Trans. Mean Image2", fontsize=12)
plt.axis('off')
plt.tight_layout()
plt.show()
"""plt.subplot(234)
plt.imshow(meanImg1-meanImg2_t, cmap='Spectral_r')
plt.colorbar(orientation='horizontal')
plt.axis('off')
plt.subplot(235)
plt.imshow(meanImg2-meanImg2_t, cmap='Spectral_r')
plt.colorbar(orientation='horizontal')
plt.axis('off')
plt.subplot(236)
plt.imshow(meanImg2-meanImg2_t_a, cmap='Spectral_r')
plt.colorbar(orientation='horizontal')
plt.axis('off')"""


# In[8]:


def compare_images(img1, img2):
    # normalize to compensate for exposure difference, this may be unnecessary
    # consider disabling it
    img1 = normalize(img1)
    img2 = normalize(img2)
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = np.sum(abs(diff))  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)

def normalize(arr):
    rng = arr.max()-arr.min()
    amin = arr.min()
    return (arr-amin)*255/rng

def mse(img1, img2):
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img2.shape[0] * img1.shape[1])
    return err

print("Image Comparison: ")
print("Image1 and Image2: ")
n_m1, n_01 = compare_images(meanImg1, meanImg2)
print("Manhattan norm:", n_m1, "/ per pixel:", n_m1/meanImg1.size)
print("Zero norm:", n_01, "/ per pixel:", n_01*1.0/meanImg1.size)
print("MSE: ", str(mse(meanImg1, meanImg2)))

print('\n\n')
print("Image1 and Image2 transformed: ")
n_m2, n_02 = compare_images(meanImg1, meanImg2_t)
print("Manhattan norm:", n_m2, "/ per pixel:", n_m2/meanImg1.size)
print("Zero norm:", n_02, "/ per pixel:", n_02*1.0/meanImg1.size)
print("MSE: ", str(mse(meanImg1, meanImg2_t)))

print('\n\n')
print("Image2 and Image2 transformed: ")
n_m3, n_03 = compare_images(meanImg2, meanImg2_t)
print("Manhattan norm:", n_m3, "/ per pixel:", n_m3/meanImg2.size)
print("Zero norm:", n_03, "/ per pixel:", n_03*1.0/meanImg2.size)
print("MSE: ", str(mse(meanImg2, meanImg2_t)))


# In[33]:


# now we create the segment mask maps for each session and apply the transform on the segment masks
stat1 = np.load('stat1.npy',allow_pickle = True)
iscell1 = np.load('iscell1.npy',allow_pickle = True)
stat2 = np.load('stat2.npy',allow_pickle = True)
iscell2 = np.load('iscell2.npy',allow_pickle = True)

#stat tupule for cells only
stat1 = stat1[iscell1[:,0]==1]
stat2 = stat2[iscell2[:,0]==1]


# In[35]:


#create a segment mask map for the primary session
numcells1 = len(stat1) #number of cells from session1
dim1 = meanImg1.shape
mask1 = np.zeros((int(dim1[0]),int(dim1[1]),3))
for n in range(numcells1):
    mask1[stat1[n]['ypix'],stat1[n]['xpix'],0] = 1

#create a segment mask map for the secondary session
numcells2 = len(stat2) #number of cells from session1
dim2 = meanImg2.shape
mask2 = np.zeros((int(dim2[0]),int(dim2[1]),3))
for n in range(numcells2):
    mask2[stat2[n]['ypix'],stat2[n]['xpix'],1] = 2
    
# get transformed mask2 image for projective and affine transformation
mask2_t_proj = warp(mask2, model_proj.inverse, cval=-1)
#mask2_t_affine = warp(mask2, model_affine.inverse, cval=-1)


# In[36]:


merged_roi_mask1 = np.zeros((meanImg1.shape[0],meanImg1.shape[1]))
merged_roi_mask1 = mask1 + mask2_t_proj

#merged_roi_mask2 = np.zeros((meanImg1.shape[0],meanImg1.shape[1]))
#merged_roi_mask2 = mask1 + mask2_t_affine

plt.figure(figsize=(8,4))
plt.subplot(111)
plt.gray()
plt.imshow(merged_roi_mask1)
plt.axis('off')
plt.title("Projective transform")
#plt.subplot(122)
#plt.gray()
#plt.imshow(merged_roi_mask2)
#plt.axis('off')
#plt.title("Affine transform")
plt.show()


# In[42]:


def calcCentroid(ypix, xpix):
    centroidY, centroidX = np.mean(ypix), np.mean(xpix)
    return centroidY, centroidX

centroidX_s1 = []
centroidY_s1 = []
for n in range(numcells1):
    centroidY, centroidX = calcCentroid(stat1[n]['ypix'],stat1[n]['xpix'])
    centroidY_s1.append(centroidY)
    centroidX_s1.append(centroidX)
    
centroidX_s2 = []
centroidY_s2 = []
for n in range(numcells2):
    centroidY, centroidX = calcCentroid(stat2[n]['ypix'],stat2[n]['xpix'])
    centroidY_s2.append(centroidY)
    centroidX_s2.append(centroidX)


# In[43]:

plt.figure()
plt.scatter(centroidX_s1, centroidY_s1, c='r', s=5)
plt.scatter(centroidX_s2, centroidY_s2, c='b', s=5)
plt.show()