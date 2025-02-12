import numpy as np
from scipy import ndimage
from sklearn.neighbors import KernelDensity
from sklearn.cluster import MeanShift
import pandas as pd

def kde2D(x, y, bandwidth, xbins=1000j, ybins=1000j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 1000x1000)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)
    
def sharpened(Z, Z_smooth):
    """Perform unsharp masking."""
    return Z-Z_smooth
    
def grad(sharpened):
    """Measure gradient using a Sobel kernel."""
    dx = ndimage.sobel(sharpened,0,output=np.float64) # horizontal derivative
    dy = ndimage.sobel(sharpened,1,output=np.float64)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag = mag / mag.max() * 255
    theta = np.degrees(np.arctan2(dy,dx))
    return mag, theta, dx, dy
    
# thin edges to 1 pixel width
# check if pixel is is maximum along gradient direction

def NonMaxSupWithInterpol(Gmag, Grad, Gx, Gy):
    """Perform non-maximum suppression with interpolation."""
    #Gmag = magnitude of gradient
    #Grad = direction of gradient
    #Gx = x component of gradient
    #Gy = y component of gradient
    
    NMS = np.zeros(Gmag.shape)
    
    for i in range(1, int(Gmag.shape[0]) - 1):
        for j in range(1, int(Gmag.shape[1]) - 1):
            if((Grad[i,j] >= 0 and Grad[i,j] <= 45) or (Grad[i,j] < -135 and Grad[i,j] >= -180)):
                yBot = np.array([Gmag[i,j+1], Gmag[i+1,j+1]])
                yTop = np.array([Gmag[i,j-1], Gmag[i-1,j-1]])
                if Gmag[i,j]==0:
                    x_est=0
                else:
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 45 and Grad[i,j] <= 90) or (Grad[i,j] < -90 and Grad[i,j] >= -135)):
                yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j+1]])
                yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j-1]])
                if Gmag[i,j]==0:
                    x_est=0
                else:
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 90 and Grad[i,j] <= 135) or (Grad[i,j] < -45 and Grad[i,j] >= -90)):
                yBot = np.array([Gmag[i+1,j] ,Gmag[i+1,j-1]])
                yTop = np.array([Gmag[i-1,j] ,Gmag[i-1,j+1]])
                if Gmag[i,j]==0:
                    x_est=0
                else:
                    x_est = np.absolute(Gx[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
            if((Grad[i,j] > 135 and Grad[i,j] <= 180) or (Grad[i,j] < 0 and Grad[i,j] >= -45)):
                yBot = np.array([Gmag[i,j-1] ,Gmag[i+1,j-1]])
                yTop = np.array([Gmag[i,j+1] ,Gmag[i-1,j+1]])
                if Gmag[i,j]==0:
                    x_est=0
                else:
                    x_est = np.absolute(Gy[i,j]/Gmag[i,j])
                if (Gmag[i,j] >= ((yBot[1]-yBot[0])*x_est+yBot[0]) and Gmag[i,j] >= ((yTop[1]-yTop[0])*x_est+yTop[0])):
                    NMS[i,j] = Gmag[i,j]
                else:
                    NMS[i,j] = 0
    
    return NMS
    
def ThreshHyst(img,lowThreshold,highThreshold):
    """Perform double threshold hysteresis."""
    img1 = np.copy(img)
    h = int(img1.shape[0])
    w = int(img1.shape[1])
    
    #if strong edge, make and edge, if below threshold, not edge.
    #if weak edge connected to strong edge, is edge.
    for i in range(1,h-1):
        for j in range(1,w-1):
            if img1[i,j] >= highThreshold:
                img1[i,j] = 255
            elif img1[i,j] < lowThreshold:
                img1[i,j] = 0
            elif ((img1[i,j] >= lowThreshold) & (img1[i,j]<highThreshold)):
                if ((img1[i-1,j-1] > highThreshold) or (img1[i-1,j] > highThreshold) or
                (img1[i-1,j+1] > highThreshold) or (img1[i,j-1] > highThreshold) or
                (img1[i,j+1] > highThreshold) or (img1[i+1,j-1] > highThreshold) or
                (img1[i+1,j] > highThreshold) or (img1[i+1,j+1] > highThreshold)):
                    img1[i,j] = 255
                else:
                    img1[i,j] = 0
    return img1

def get_range(threshold, sigma=0.33):
    """Get threshold ranges."""
    return (1-sigma)*threshold,(1+sigma)*threshold

def get_labels(hyst_img):
    """Get cluster labels."""
    edges = np.transpose(np.where(hyst_img==255))

    ms = MeanShift(bandwidth=11, bin_seeding=True)
    segments_ms = ms.fit(edges)
    labels = segments_ms.labels_
    return labels, edges

def convert_to_wcs(labels,X,Y,edges,wcs_275):
    """Convert label coordinates to xy and wcs coordinates."""
    xy_arr = []

    for i in np.unique(labels):
        xy = np.where(labels==i)
        xy_arr.append([X[np.transpose(edges[xy])[0],np.transpose(edges[xy])[1]],Y[np.transpose(edges[xy])[0],np.transpose(edges[xy])[1]]])

    wcs_coords = []
    for i in np.arange(len(xy_arr)):
        wcs_coords.append(wcs_275.pixel_to_world(xy_arr[i][0],xy_arr[i][1]))
    return xy_arr, wcs_coords

def make_dataframe(xy_arr,wcs_coords):
    """Build dataframe of each cluster with their xy coordinates, wcs coordinates, and labels."""
    for i in np.arange(len(xy_arr)):
        temp = pd.DataFrame(np.transpose([xy_arr[i][0],xy_arr[i][1],
                                          wcs_coords[i].ra.deg,wcs_coords[i].dec.deg,
                                          np.zeros(len(wcs_coords[i].ra))+i]),columns=['x','y','ra','dec','label'])
        if i == 0:
            df2 = temp
        if i>0:
            df2 = pd.concat([df2,temp])
    return df2
