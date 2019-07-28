
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import numpy as np
import cv2
import detect_edges_image
import detect_clusters_image


import h5py




def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)

	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, 0, 50,50)

	# return the edged image
	return edged

resize=True

raw_image = cv2.imread('./images/NSCLC_8-Plex_WSI_Ultivue.jpg')
if resize: raw_image = cv2.resize(raw_image,(100,100))
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HSV)

print (type(hsv_image))


if resize:
    plt.imshow(raw_image)
    plt.show()

    print ("Normalizing Pixel colours..")

    pixel_colors = raw_image.reshape((np.shape(raw_image)[0]*np.shape(raw_image)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    h, s, v = cv2.split(hsv_image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    #plt.show()

mask1 = cv2.inRange(hsv_image, (0, 30, 10), (10, 255, 255))
mask2 = cv2.inRange(hsv_image, (15, 30, 10), (30, 255, 255))
mask3 = cv2.inRange(hsv_image, (70, 30, 10), (90, 255, 255))
mask3 = cv2.inRange(hsv_image, (70, 30, 10), (90, 255, 255))
mask4 = cv2.inRange(hsv_image, (90, 30, 10), (120, 255, 255))
mask5 = cv2.inRange(hsv_image, (120, 0, 10), (160, 255, 255))

result1 = cv2.bitwise_and(raw_image, raw_image, mask=mask1)
print ('Mask 1 Completed')

result2 = cv2.bitwise_and(raw_image, raw_image, mask=mask2)
print ('Mask 2 Completed')

result3 = cv2.bitwise_and(raw_image, raw_image, mask=mask3)
print ('Mask 3 Completed')

result4 = cv2.bitwise_and(raw_image, raw_image, mask=mask4)
print ('Mask 4 Completed')

result5 = cv2.bitwise_and(raw_image, raw_image, mask=mask5)
print ('Mask 5 Completed')

detect_clusters_image.tf_image_to_points(result4)

segment_scale_factor = 20

hed, canny = detect_edges_image.HED_auto(raw_image,20)
hed1, canny = detect_edges_image.HED_auto(result1.copy(),segment_scale_factor)
hed2, canny = detect_edges_image.HED_auto(result2.copy(),segment_scale_factor)
hed3, canny = detect_edges_image.HED_auto(result3.copy(),segment_scale_factor)
hed4, canny = detect_edges_image.HED_auto(result4.copy(),segment_scale_factor)
hed5, canny = detect_edges_image.HED_auto(result5.copy(),segment_scale_factor)

plt.subplot(5, 2, 1)
plt.imshow(raw_image)
plt.subplot(5, 2, 2)
plt.imshow(hed)

plt.subplot(5, 2, 3)
plt.imshow(result1)
plt.subplot(5, 2, 4)
plt.imshow(hed1)
plt.subplot(5, 2, 5)
plt.imshow(result2)
plt.subplot(5, 2, 6)
plt.imshow(hed2)
plt.subplot(5, 2, 7)
plt.imshow(result4)
plt.subplot(5, 2, 8)
plt.imshow(hed4)
plt.subplot(5, 2, 9)
plt.imshow(result5)
plt.subplot(5, 2, 10)
plt.imshow(hed5)

plt.show()

f = h5py.File('image_data_post.hdf5', 'w')
dset = f.create_dataset("raw_image", data=raw_image, chunks=True)
dset = f.create_dataset("hed1", data=hed1, chunks=True)
dset = f.create_dataset("hed2", data=hed2, chunks=True)
dset = f.create_dataset("hed3", data=hed3, chunks=True)
dset = f.create_dataset("hed4", data=hed4, chunks=True)
dset = f.create_dataset("hed5", data=hed5, chunks=True)
