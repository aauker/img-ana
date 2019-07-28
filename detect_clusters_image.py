print ('[INFO] Initialize Clustering)')

from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle




def tf_image_to_points(image):
    print (image.shape)
    indices = np.where(image > [10])
    coordinates = zip(indices[1], -indices[0])
    print (len(coordinates))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.subplot(1,2,2)
    plt.scatter(indices[1],-indices[0])



    af = AffinityPropagation(preference=-100, damping=0.9).fit(coordinates)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_

    print (len(labels))

    n_clusters_ = len(cluster_centers_indices)

    print('Estimated number of clusters: %d' % n_clusters_)
    # Plot result

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = coordinates[cluster_centers_indices[k]]
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        for i,x in enumerate(coordinates):
            if k==labels[i]: plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]],col)

    plt.show()
