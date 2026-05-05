import cv2
import numpy as np
import matplotlib.pyplot as plt
import functions_file

# Load the image in grayscale directly
image_left = cv2.imread('im1.png', cv2.IMREAD_GRAYSCALE)
image_right = cv2.imread('im2.png', cv2.IMREAD_GRAYSCALE)

# Normalize the values to the [0, 1] range
#image_left = image_left.astype(np.float32) / 255.0
#image_right = image_right.astype(np.float32) / 255.0

data = np.load("knownpoints.npy", allow_pickle=True)
# Descriptor Parameters
rhom = 5
rhoM = 20
rhostep = 1
N = 8
# Camera Parameters
D = 144.049 #mm
f = 4396.869 #px
doffs = 185.788 #px

print(data.shape[0])

depth_map = functions_file.verifyResults(data, image_left.shape,
                    image_left, image_right, False, D, f, doffs)

#functions_file.compareDetectors(image_left, depth_map)



#print("--- Computing Depth and Disparity Map ---")
#[disparity_map, depth_map] = computeDepthMap(image_left, image_right, f, D, doffs, use_upgrade=False)