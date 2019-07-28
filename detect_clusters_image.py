print ('[INFO] Initialize Clustering)')

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from scipy.optimize import curve_fit
import sys


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import cv2
import random
from matplotlib import cm
from matplotlib import colors

def updt(total, progress):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()

def expo_funct(x, a, b, c):
    return a*np.exp(-b*x) + c

def tf_image_to_points(inp_image):

    f, ax = plt.subplots()
    image = cv2.resize(inp_image,(500,500))
    indices = np.where(image > [10])
    (H, W) = image.shape[:2]
    coordinates_raw = zip(indices[1]/float(W), -indices[0]/float(H))
    if len(coordinates_raw) > 100000: coordinates = random.sample(coordinates_raw, 100000)
    else: coordinates = coordinates_raw

    print ('[INFO] Reduction Fraction: = {0}%'.format(100.0*len(coordinates)/ len(coordinates_raw)))
    print ('[INFO] Input size = {0}'.format(len(coordinates)))



    l_nclusters = []
    wcss = []
    curvature = []
    l_nclusters_curvature = []

    first = 10
    last = 100

    print ('[INFO] Optimizing Cluster Number')

    for n_clust in range(first,last,10):
        updt(last-first, n_clust -first + 1)
        af = MiniBatchKMeans(n_clusters=n_clust, random_state=0).fit(coordinates)
        l_nclusters.append(n_clust)
        wcss.append (af.inertia_)

    #normalize
    max_val = max(wcss)
    wcss = wcss/max_val

    popt, pcov = curve_fit(expo_funct,l_nclusters,wcss)

    #un normalize
    wcss = wcss * max_val
    popt[0] = popt[0]*max_val
    popt[2] = popt[2]*max_val

    t = np.linspace(first, last ,1000)
    y_fit = expo_funct(t,*popt)
    for i in range(1, len(y_fit)-1):
        dt = t[i]-t[i-1]
        f_first_derv = (y_fit[i]- y_fit[i-1]) / (dt)
        f_second_derv = (y_fit[i+1] -  2.0*y_fit[i] +  y_fit[i-1] ) / dt**2
        #curve = abs(f_second_derv) /  ( 1 + f_second_derv**2)**(1.5)
        curve = abs(f_second_derv) /  (( 1 + f_first_derv**2)**(1.5))
        curvature.append(curve*max_val)
        l_nclusters_curvature.append(t[i])


    #af = AffinityPropagation(preference=-2, damping=0.8).fit(coordinates)
    #cluster_centers_indices = af.cluster_centers_indices_

    index_min = int(t[np.argmax(curvature)])
    print('[INFO] Optimial number of clusters: %d' % index_min)
    plt.plot(t,expo_funct(t,*popt), linewidth=2, color='magenta', label='Expo Chi Squared Fit')
    plt.plot(l_nclusters,wcss)
    plt.plot(l_nclusters_curvature,curvature)
    #plt.yscale('log')
    plt.show()

    af = MiniBatchKMeans(n_clusters=index_min, random_state=0).fit(coordinates)



    plt.subplot(1,1,1)
    plt.imshow(image)

    cluster_centers_indices = af.cluster_centers_
    labels = af.labels_

    n_clusters_ = len(cluster_centers_indices)

    # Plot result
    cmap = cm.rainbow(np.linspace(0,1,n_clusters_))
    print ('[INFO] Plotting clusters...')


    for k in range(n_clusters_):
        col = colors.to_hex(cmap[k])
        class_members = labels == k
        #cluster_center = coordinates[cluster_centers_indices[k]]
        cluster_center = cluster_centers_indices[k]
        plt.plot(cluster_center[0]*W, -cluster_center[1]*H, 'o', markerfacecolor=col, markeredgecolor=col, markersize=5)
        cluster_points_x = [coord[0]*W for coord,lab in zip(coordinates,labels) if k==lab ]
        cluster_points_y = [coord[1]*-H for coord,lab in zip(coordinates,labels) if k==lab ]
        plt.scatter(cluster_points_x, cluster_points_y, facecolors='none', edgecolors=col, linewidths=0.05, marker='s')

    plt.show()
