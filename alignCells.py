# -*- coding: utf-8 -*-
"""
Created on Thu May 20 23:11:10 2021

@author: Rajat
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform
from skimage.feature import (plot_matches, match_descriptors, ORB)
from skimage.transform import warp


def transformImages(ops1,ops2,stat1,stat2,c1,c2,transformation='projective',plotDat=False):
    img1 = np.array(ops1['meanImgE'], dtype=np.double)
    img2 = np.array(ops2['meanImgE'], dtype=np.double)
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
    if plotDat:
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4,8))
        plt.gray()
        plot_matches(ax[0], img1, img2, keypoints1, keypoints2, matches12)
        ax[0].set_title('All correspondences')
        ax[0].axis('off')
        inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(ax[1], img1, img2, matches_img1, matches_img2, np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
        ax[1].set_title('Correct correspondences')
        ax[1].axis('off')
        outlier_idxs = np.nonzero(outliers)[0]
        plot_matches(ax[2], img1, img2, matches_img1, matches_img2, np.column_stack((outlier_idxs, outlier_idxs)), matches_color='r')
        ax[2].set_title('Faulty correspondences')
        ax[2].axis('off')
        plt.tight_layout()
        plt.show()
    
    # warp image1 wrt image2
    img2_transformed = warp(img2, model_robust.inverse, cval=-1)
    

    #create a segment mask map for the primary session
    numcells1 = len(stat1) #number of cells from session1
    dim1 = img1.shape
    mask1 = np.zeros((int(dim1[0]),int(dim1[1]),3))
    rgb = c1
    for n in range(numcells1):
        mask1[stat1[n]['ypix'],stat1[n]['xpix'],0] = rgb[0]
        mask1[stat1[n]['ypix'],stat1[n]['xpix'],1] = rgb[1]
        mask1[stat1[n]['ypix'],stat1[n]['xpix'],2] = rgb[2]
    
    #create a segment mask map for the secondary session
    numcells2 = len(stat2) #number of cells from session1
    dim2 = img2.shape
    mask2 = np.zeros((int(dim2[0]),int(dim2[1]),3))
    rgb = c2
    for n in range(numcells2):
        mask2[stat2[n]['ypix'],stat2[n]['xpix'],0] = rgb[0]
        mask2[stat2[n]['ypix'],stat2[n]['xpix'],1] = rgb[1]
        mask2[stat2[n]['ypix'],stat2[n]['xpix'],2] = rgb[2]
    
    # get transformed mask2 image for projective and affine transformation
    mask2_t_proj = warp(mask2, model_robust.inverse, cval=-1)

    return img1, img2_transformed, mask1, mask2_t_proj 

# In[5]:


# load the data and mean images
ops1 = np.load('opsDay1.npy', allow_pickle=True).item()
ops2 = np.load('opsDay2.npy', allow_pickle=True).item()
ops3 = np.load('opsDay3.npy', allow_pickle=True).item()
ops4 = np.load('opsDay4.npy', allow_pickle=True).item()
ops5 = np.load('opsDay5.npy', allow_pickle=True).item()
stat1 = np.load('statDay1.npy',allow_pickle = True)
stat2 = np.load('statDay2.npy',allow_pickle = True)
stat3 = np.load('statDay3.npy',allow_pickle = True)
stat4 = np.load('statDay4.npy',allow_pickle = True)
stat5 = np.load('statDay5.npy',allow_pickle = True)

# get transformed masks
img1, img2, mask1, mask2 = transformImages(ops1, ops2, stat1, stat2, [0.3,0,0], [0,0.3,0], 'projective')
img1, img3, mask1, mask3 = transformImages(ops1, ops3, stat1, stat3, [0.3,0,0], [0,0.3,0.3],'projective')
img1, img4, mask1, mask4 = transformImages(ops1, ops4, stat1, stat4,[0.3,0,0], [0.3,0,0.3],'projective')
img1, img5, mask1, mask5 = transformImages(ops1, ops5, stat1, stat5, [0.3,0,0], [0.3,0.3,0],'projective')

# plot the raw data
plt.figure()
plt.subplot(351)
plt.imshow(img1, cmap='gray')
plt.title("Session1", fontsize=12)
plt.axis('off')
plt.subplot(352)
plt.imshow(img2, cmap='gray')
plt.title("Session2", fontsize=12)
plt.axis('off')
plt.subplot(353)
plt.imshow(img3, cmap='gray')
plt.title("Session3", fontsize=12)
plt.axis('off')
plt.subplot(354)
plt.imshow(img4, cmap='gray')
plt.title("Session4", fontsize=12)
plt.axis('off')
plt.subplot(355)
plt.imshow(img5, cmap='gray')
plt.title("Session5", fontsize=12)
plt.axis('off')
plt.subplot(356)
plt.imshow(mask1)
plt.axis('off')
plt.subplot(357)
plt.imshow(mask2)
plt.axis('off')
plt.subplot(358)
plt.imshow(mask3)
plt.axis('off')
plt.subplot(359)
plt.imshow(mask4)
plt.axis('off')
plt.subplot(3,5,10)
plt.imshow(mask5)
plt.axis('off')
plt.subplot(3,5,11)
plt.imshow(mask1)
plt.axis('off')
plt.subplot(3,5,12)
plt.imshow(mask1+mask2)
plt.axis('off')
plt.subplot(3,5,13)
plt.imshow(mask1+mask2+mask3)
plt.axis('off')
plt.subplot(3,5,14)
plt.imshow(mask1+mask2+mask3+mask4)
plt.axis('off')
plt.subplot(3,5,15)
plt.imshow(mask1+mask2+mask3+mask4+mask5)
plt.axis('off')
#plt.tight_layout()
plt.show
