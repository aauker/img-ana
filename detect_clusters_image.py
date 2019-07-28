print ('[INFO] Initialize Clustering)')

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import cv2
import random
from matplotlib import cm
from matplotlib import colors


def tf_image_to_points(inp_image):
    f, ax = plt.subplots()
    image = cv2.resize(inp_image,(100,100))
    print (image.shape)
    indices = np.where(image > [200])
    (H, W) = image.shape[:2]
    coordinates_raw = zip(indices[1]/float(W), -indices[0]/float(H))
    coordinates = coordinates_raw
    #coordinates = random.sample(coordinates_raw, 2000)

    print ('[INFO] Reduction Fraction: = {0}%'.format(100.0*len(coordinates)/ len(coordinates_raw)))
    print ('[INFO] Input size = {0}'.format(len(coordinates)))
    plt.subplot(1,1,1)
    plt.imshow(image)

    af = AffinityPropagation(damping=0.5).fit(coordinates)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    print (len(labels))

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    # Plot result
    cmap = cm.rainbow(np.linspace(0,1,n_clusters_))

    for k in range(n_clusters_):
        col = colors.to_hex(cmap[k])
        #col = 'r'
        class_members = labels == k
        cluster_center = coordinates[cluster_centers_indices[k]]
        print (col)
        plt.plot(cluster_center[0]*W, -cluster_center[1]*H, 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for i,x in enumerate(coordinates):
            if k==labels[i]:
                print (col)
                plt.plot([cluster_center[0]*W, x[0]*W], [-cluster_center[1]*H, -x[1]*H],col)

    plt.show()
