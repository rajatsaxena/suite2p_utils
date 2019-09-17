#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import disk
from skimage.morphology import (erosion, dilation, opening, closing)


# In[2]:


def clipROIs(opsfilename, statfilename, iscellfilename):
    # load the ops file 
    ops = np.load(opsfilename, allow_pickle=True).item()
    stat = np.load(statfilename, allow_pickle=True)
    iscell = np.load(iscellfilename,allow_pickle = True)
    # stat tuple for cells only
    # stat = stat[iscell[:,0]==1]
    # load the mean image
    meanImg = ops['meanImg']

    # create a segment mask map for the primary session
    ncells = len(stat) 
    # iterate through the number of cells from session1
    for n in range(ncells):
        mask = np.zeros((int(meanImg.shape[0]),int(meanImg.shape[1])))
        mask[stat[n]['ypix'],stat[n]['xpix']] = 1
        x,y = np.nonzero(mask)
        # get the cell roi mask
        if min(x)-5>0 and max(x)+5<mask.shape[0] and min(y)-5>0 and max(y)+5<mask.shape[1]:
            mask = mask[min(x)-5:max(x)+5,min(y)-5:max(y)+5]
            
            plt.figure(figsize=(2,2))
            # close the small openings
            mask = closing(mask, disk(2))
            contours = measure.find_contours(mask, 0.8)
            for n, contour in enumerate(contours):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, c='k')

            # erosion followed by dilation
            mask_opening = opening(mask, disk(2))
            contours_opening = measure.find_contours(mask_opening, 0.8)
            for n, contour in enumerate(contours_opening):
                plt.plot(contour[:, 1], contour[:, 0], linewidth=2, c='r')
                
            plt.show()


# In[3]:


opsfilename = 'ops3.npy'
statfilename = 'stat3.npy'
iscellfilename = 'iscell3.npy'

clipROIs(opsfilename, statfilename, iscellfilename)

