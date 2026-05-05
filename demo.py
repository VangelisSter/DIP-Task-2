import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions_file import computeDisparity, computeDepth, computeDepthMap, myDetectHarrisFeatures

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

print(data.shape)
print("--- Running Sanity Check on Known Points ---")
    
    # NOTE: Because I do not know the exact shape of your .npy file, 
    # I am assuming it is a list of arrays formatted like:
    # [x_Left, y_Left, x_right, y_right, d_est, d_gt]
    # You will likely need to adjust these indices based on how the professor saved it!
    
for row in data:
    p_L = (row[0], row[1])
    p_R = (row[2], row[3])
    d_est = row[4]
    d_gt = row[5]
        
    # Calculate our values
    calc_d = computeDisparity(p_L, p_R)
    calc_z = computeDepth(calc_d, f, D, doffs)
    #calc_z = computeDepth(calc_d, f, D, doffs)
        
        # Compare
    #print(f"Points x: {p_L[0], p_R[0]}:")
    print(f"  Disparity -> Calculated: {calc_d:.3f} | Estimated from Algorithm: {d_est:.3f} | Ground Truth: {d_gt:.3f}")
    print("-" * 30)

print("--- Checking myHarrisDetectFeatures ---")
# Detect corners
corners = cv2.cornerHarris(image_left, blockSize=2, ksize=3, k=0.04)
# Normalize
corners = cv2.normalize(corners, None)
ret, corners = cv2.threshold(corners, 0.01, 1, cv2.THRESH_BINARY)
# Get corner coordinates
corner_coords = np.where(corners > 0)

corner_coords_correct = np.array([corner_coords[0], corner_coords[1]])

print(f"OpenCV Harris: {corner_coords_correct.T}, length: {corner_coords_correct.T.shape}")

cv2_coords = corner_coords_correct.T

my_corners = myDetectHarrisFeatures(image_left)
custom_coords = my_corners

print(f"myDetectHarrisFeatures: {my_corners}, length: {my_corners.shape}")

plt.figure(figsize=(14, 7))

    # --- Plot 1: Your Custom Detector ---
plt.subplot(1, 2, 1)
plt.imshow(image_left, cmap='gray')
    # Note: matplotlib scatter uses (x, y), so we pass column (index 1) then row (index 0)
if len(custom_coords) > 0:
    plt.scatter(custom_coords[:, 1], custom_coords[:, 0], 
                c='red', s=20, marker='x', label='Custom')
plt.title(f"My Detector ({len(custom_coords)} points)")
plt.legend()
plt.axis('off')

    # --- Plot 2: OpenCV Detector ---
plt.subplot(1, 2, 2)
plt.imshow(image_left, cmap='gray')
if len(cv2_coords) > 0:
    plt.scatter(cv2_coords[:, 1], cv2_coords[:, 0], 
                c='cyan', s=20, marker='+', label='OpenCV')
plt.title(f"OpenCV ({len(cv2_coords)} points)")
plt.legend()
plt.axis('off')

    # Display the visualization
plt.tight_layout()
plt.show()

#print("--- Computing Depth and Disparity Map ---")
#[disparity_map, depth_map] = computeDepthMap(image_left, image_right, f, D, doffs, use_upgrade=False)