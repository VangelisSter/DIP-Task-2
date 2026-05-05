import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import maximum_filter, gaussian_filter

# Set up global cache variables
_cached_image_id = None
_cached_R_matrix = None
_cached_local_max_R = None

def myLocalDescriptor(I: np.ndarray, p: tuple, rhom: float, 
                      rhoM: float, rhostep: float, N: int) -> np.ndarray:
    K, M = I.shape
    # K rows, M columns

    p_array = np.array(p)

    # Out of Bounds check
    empty = (p_array > [K, M]).any or (p_array < rhoM).any
    if empty:
        return np.empty(0)
    
    #rho = [i for i in range(rhom, rhoM, rhostep)]
    radii = np.arange(rhom, rhoM + rhostep, rhostep)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    descriptor = []

    for r in radii:
        # x = p1 + r * cos(theta)
        [row_exact, col_exact] = p_array + r * [sin_theta, cos_theta]
        
        # We round down to get the exact pixel
        col_base = np.floor(col_exact).astype(int)
        row_base = np.floor(row_exact).astype(int)


        # Find neighbouring pixels while minding the boundaries
        col_next = np.clip(col_base + 1, 0, M - 1)
        col_before = np.clip(col_base - 1, 0, M - 1)
        row_next = np.clip(row_base + 1, 0, K - 1)
        row_before = np.clip(row_base - 1, 0, K - 1)

        # Get the actual pixel intensity values from the image I
        I_top =    I[row_before, col_base]       # Top
        I_bottom = I[row_next, col_base]       # Bottom
        I_left =   I[row_base, col_before]       # Left
        I_right =  I[row_base, col_next]       # Right
    
        # First find the mean of the neighbours
        neigh_mean = (I_top + I_bottom + I_left + I_right) / 4.0

        # Then find the mean of the N values regarding this radius
        descriptor.append(np.mean(neigh_mean))
    
    return np.array(descriptor)

def myLocalDescriptorUpgrade(I: np.ndarray, p: tuple, rhom: float, 
                      rhoM: float, rhostep: float, N: int) -> np.ndarray:
    
    K, M = I.shape
    # K rows, M columns

    p_array = np.flip(np.array(p)) # To transform from row,col to x,y

    # Out of Bounds check
    empty = (p_array > [K, M]).any or (p_array < rhoM).any
    if empty:
        return np.empty(0)
    
    #rho = [i for i in range(rhom, rhoM, rhostep)]
    radii = np.arange(rhom, rhoM + rhostep, rhostep)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    desc_means = []
    desc_stds = []

    for r in radii:
        # x = p1 + r * cos(theta)
        [row_exact, col_exact] = p_array + r * [sin_theta, cos_theta]
        
        # Find the coordinates for the Top-Left pixel
        col_left = np.clip(np.floor(col_exact).astype(int), 0, M - 1)
        row_top = np.clip(np.floor(row_exact).astype(int), 0, K - 1)

        # Find the Right and Bottom coordinates
        col_right = np.clip(col_left + 1, 0, M - 1)
        row_bottom = np.clip(row_top + 1, 0, K - 1)

        # Calculate the fractional distances
        dx = col_exact - col_left
        dy = row_exact - row_top

        # Get the actual pixel intensity values from the image I
        I_top_left = I[row_top, col_left]
        I_top_right = I[row_top, col_right]
        I_bottom_left = I[row_bottom, col_left]
        I_bottom_right = I[row_bottom, col_right]


        # Apply the bilinear interpolation formula
        interpolated_pixels = (
            I_top_left * (1 - dx) * (1 - dy) +
            I_top_right * dx * (1 - dy) +
            I_bottom_left * (1 - dx) * dy +
            I_bottom_right * dx * dy
        )

        # Calculate the feature stats for this specific ring
        desc_means.append(np.mean(interpolated_pixels)) # Ring mean
        desc_stds.append(np.std(interpolated_pixels)) # Ring STD

    # Combine them into a single, raw fingerprint vector
    # If N=3 rings, this array now has 6 elements [m1, m2, m3, s1, s2, s3]
    raw_desc_array = np.concatenate((desc_means, desc_stds))

    # Apply Illumination Invariance (Z-score Normalization)
    # We calculate the mean and Vector STD of the ENTIRE 6-element array
    vector_mean = np.mean(raw_desc_array)
    vector_std = np.std(raw_desc_array)

    # Normalize the entire array at once
    normalized_desc = (raw_desc_array - vector_mean) / (vector_std + 1e-7)
    
    return normalized_desc


def isCorner(I: np.ndarray, p: tuple, k: float, Rthres: float) -> bool:
    global _cached_image_id, _cached_R_matrix, _cached_local_max_R

    #Exctract the points and dimensions
    #p1, p2 = p
    #K, N = I.shape
    # K rows, N columns

    # We make the assumption the weight uniform, as in in the neighbour of p it is 1 and elsewere 0

    # First we need the neighbouring pixels
    # We use +2 for the ends because Python slicing is exclusive at the upper bound
    #r_start = max(0, p1 - 2)
    #r_end   = min(K, p1 + 3) 
    #c_start = max(0, p2 - 2)
    #c_end   = min(N, p2 + 3)

    # 2. Extract the 5x5 patch
    #patch = I[r_start:r_end, c_start:c_end]


    # Calculate the derivative arrays
    #Iy, Ix = np.gradient(patch)

    # Calculate the M matrix components
    #M11 = np.sum(Ix ** 2)
    #M12 = np.sum(Ix * Iy)
    #M22 = np.sum(Iy ** 2)

    # Calculate the determinent and trace

    #det_M = (M11 * M22) - (M12 ** 2)
    #trace_M = M11 + M22

    # Calculate R
    #R = det_M - k * (trace_M ** 2)

    #return R > Rthres
    # 1. THE CACHE CHECK
    # id(I) gets the unique memory address of the image.
    # If this is a brand new image we haven't seen yet, compute the math ONCE.
    if id(I) != _cached_image_id:
        _cached_image_id = id(I)
        
        # Fast Vectorized Math for the entire image (Runs exactly once)
        Iy, Ix = np.gradient(I)
        #Ixx_pad = np.pad(Ix ** 2, 1, mode='constant')
        #Iyy_pad = np.pad(Iy ** 2, 1, mode='constant')
        #Ixy_pad = np.pad(Ix * Iy, 1, mode='constant')
        
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        #Sxx = sum(Ixx_pad[dy : dy + I.shape[0], dx : dx + I.shape[1]] for dy in range(3) for dx in range(3))
        #Syy = sum(Iyy_pad[dy : dy + I.shape[0], dx : dx + I.shape[1]] for dy in range(3) for dx in range(3))
        #Sxy = sum(Ixy_pad[dy : dy + I.shape[0], dx : dx + I.shape[1]] for dy in range(3) for dx in range(3))
        
        Sxx = gaussian_filter(Ixx, sigma=1.0)
        Syy = gaussian_filter(Iyy, sigma=1.0)
        Sxy = gaussian_filter(Ixy, sigma=1.0)

        # Save the fully computed R matrix to the global cache
        _cached_R_matrix = (Sxx * Syy) - (Sxy ** 2) - k * ((Sxx + Syy) ** 2)
        # This creates a map where every pixel is replaced by the maximum value in its 3x3 neighborhood
        _cached_local_max_R = maximum_filter(_cached_R_matrix, size=50)
    # 2. THE LOOKUP
    row, col = p[0], p[1]
    r_val = _cached_R_matrix[row, col]
    
    # NEW: A pixel is a corner ONLY if it beats the threshold 
    # AND its value is exactly equal to the local maximum in its 3x3 window.
    is_above_threshold = r_val > Rthres
    is_local_maximum = r_val == _cached_local_max_R[row, col]
    return is_above_threshold and is_local_maximum

def myDetectHarrisFeatures(I: np.ndarray) -> np.ndarray:
    corners = []
    K, N = I.shape
    
    # Define parameters 
    k = 0.24968
    Rthres = 1.2
    
    # Loop over the pixels
    # We skip the outermost 1-pixel border to ensure the 3x3 window 
    # of the isCorner function fits perfectly inside the image without edge distortion.
    corners = [[r, c] for r in range(1, K - 1) for c in range(1, N - 1) 
               if isCorner(I, (r, c), k, Rthres)]
                
    # Cast to an np array
    return np.array(corners)

def computeDisparity(p1: tuple, p2: tuple) -> float:

    d = p1[0] - p2[0]
    return d

def computeDepth(d: float, f: float, D: float, doffs: float) -> float:
    doffs = 185.788 #px
    d = d + doffs

    # Safety check: Prevent division by zero if disparity is exactly 0
    if d == 0:
        return float('inf')
    
    # Else calcluate the depth from the formula
    z = f * D / d   
    return z

def computeDepthMap(I_left: np.ndarray, I_right: np.ndarray,f: float, 
                    D: float, doffs: float, use_upgrade: bool) -> np.ndarray:
    
    K, M = I_left.shape
    rhom = 5
    rhoM = 20
    rhostep = 1
    N = 8

    depth_map = np.zeros((K, M))
    disparity_map = np.zeros((K, M))

    # First we need to the corners of the images
    print("Detecting corners...")
    corners_left = myDetectHarrisFeatures(I_left)
    corners_right = myDetectHarrisFeatures(I_right)

    print(f"Computing {'Upgraded' if use_upgrade else 'Basic'} descriptors...")
    if use_upgrade:
        desc_left  = [myLocalDescriptorUpgrade(I_left, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_left]
        desc_right = [myLocalDescriptorUpgrade(I_right, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_right] 
    else:
        desc_left  = [myLocalDescriptor(I_left, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_left]
        desc_right = [myLocalDescriptor(I_right, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_right]

    # Matching loop to match the left and right corners
    print("Matching points...")
    for i, p_left in enumerate(corners_left):
        current_desc_left = desc_left[i]

        best_match_idx = -1
        min_distance = float('inf')

        # Try to match to one on the right
        for j, p_right in enumerate(corners_right):

            # Since we know the images are shifted only in one direction, corners are on the same row
            if p_left[0] == p_right[0]:

                distance = np.linalg.norm(current_desc_left - desc_right[j])

                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = j   

        # If a match is found, calculate the depth
        if best_match_idx != -1:
            best_p_right = corners_right[best_match_idx]

            d = computeDisparity(p_left, best_p_right)
            z = computeDepth(d, f, D, doffs)

            depth_map[p_left[0], p_left[1]] = z
            disparity_map = d
            
    print("Processing complete.")
    return disparity_map, depth_map



def compareDetectors(I: np.ndarray) -> None:
    print("--- Checking myHarrisDetectFeatures ---")
    # Detect corners
    corners = cv2.cornerHarris(I, blockSize=2, ksize=3, k=0.04)
    # Normalize
    corners = cv2.normalize(corners, None)
    ret, corners = cv2.threshold(corners, 0.01, 1, cv2.THRESH_BINARY)
    # Get corner coordinates
    corner_coords = np.where(corners > 0)

    corner_coords = (np.array([corner_coords[0], corner_coords[1]])).T

    print(f"OpenCV Harris: {corner_coords}, length: {corner_coords.shape}")

    cv_coords = corner_coords

    custom_coords = myDetectHarrisFeatures(I)

    print(f"myDetectHarrisFeatures: {custom_coords}, length: {custom_coords.shape}")
    plt.figure(figsize=(14, 7))

    # --- Plot 1: Harris Detector ---
    plt.subplot(1, 2, 1)
    plt.imshow(I, cmap='gray')
        # Note: matplotlib scatter uses (x, y), so we pass column (index 1) then row (index 0)
    if len(custom_coords) > 0:
        plt.scatter(custom_coords[:, 1], custom_coords[:, 0], 
                    c='red', s=20, marker='x', label='Custom')
    plt.title(f"My Detector ({len(custom_coords)} points)")
    plt.legend()
    plt.axis('off')

        # --- Plot 2: OpenCV Detector ---
    plt.subplot(1, 2, 2)
    plt.imshow(I, cmap='gray')
    if len(cv_coords) > 0:
        plt.scatter(cv_coords[:, 1], cv_coords[:, 0], 
                    c='cyan', s=20, marker='+', label='OpenCV')
    plt.title(f"OpenCV ({len(cv_coords)} points)")
    plt.legend()
    plt.axis('off')

    # Display the visualization
    plt.tight_layout()
    plt.show()

def verifyResults(data: np.ndarray, shape: tuple, I_left: np.ndarray, I_right: np.ndarray,
                    use_upgrade: bool, D: float, f: float, doffs: float) -> np.ndarray:
    print("--- Running Sanity Check on Known Points ---")

    # [x_Left, y_Left, x_right, y_right, d_est, d_gt]

    K, M = shape
    rhom = 5
    rhoM = 20
    rhostep = 1
    N = 8

    depth_map = np.zeros([K, M])

    corners_left = np.zeros([data.shape[0], 2]) # As many corners
    corners_right = np.zeros_like(corners_left)

    # Exctract Corners
    for i, row in enumerate(data):
        p_L = (row[0], row[1])
        p_R = (row[2], row[3])
        d_est = row[4]
        d_gt = row[5]
        corners_left[i] = [row[1], row[0]] # Inputs is xy, we want ij
        corners_right[i] = [row[3], row[2]]
        # Calculate our values
        calc_d = computeDisparity(p_L, p_R)
        calc_z = computeDepth(calc_d, f, D, doffs)
            
        # Compare
        #print(f"Points x: {p_L[0], p_R[0]}:")
        print(f"  Disparity -> Calculated: {calc_d:.3f} | Estimated from Algorithm: {d_est:.3f} | Ground Truth: {d_gt:.3f}")
        print("-" * 30)
        depth_map[int(row[1]), int(row[0])] = calc_z
    
    # Now we have to see if we can match them as well from our descriptors
    if use_upgrade:
        desc_left  = [myLocalDescriptorUpgrade(I_left, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_left]
        desc_right = [myLocalDescriptorUpgrade(I_right, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_right] 
    else:
        desc_left  = [myLocalDescriptor(I_left, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_left]
        desc_right = [myLocalDescriptor(I_right, tuple(corner), rhom, rhoM, rhostep, N) for corner in corners_right]

    # Matrices to store matched points
    matched_corners_left = []
    matched_corners_right = []


    for i, p_left in enumerate(corners_left):
        current_desc_left = desc_left[i]

        best_match_idx = -1
        min_distance = float('inf')

        # Try to match to one on the right
        for j, p_right in enumerate(corners_right):

            # Since we know the images are shifted only in one direction, corners are on the same row
            if p_left[0] == p_right[0]:

                distance = np.linalg.norm(current_desc_left - desc_right[j])

                if distance < min_distance:
                    min_distance = distance
                    best_match_idx = j  

        if best_match_idx != -1:
            matched_corners_right.append(corners_right[best_match_idx])
            matched_corners_left.append(corners_left[i])
    
    matched_corners_right = np.array(matched_corners_right)
    print(matched_corners_right)
    matched_corners_left = np.array(matched_corners_left)
    print(matched_corners_right)

    print(matched_corners_left.shape)
    print(matched_corners_right.shape)

        # --- Plot 1: Harris Detector ---
    plt.subplot(1, 2, 1)
    plt.imshow(I_left, cmap='gray')
        # Note: matplotlib scatter uses (x, y), so we pass column (index 1) then row (index 0)
    if len(matched_corners_left) > 0:
        plt.scatter(matched_corners_left[:, 1], matched_corners_left[:, 0], 
                    c='red', s=20, marker='x', label='Corners Detected Left')
    plt.title(f"Corners Detected Left ({len(matched_corners_left)} points)")
    plt.legend()
    plt.axis('off')

        # --- Plot 2: OpenCV Detector ---
    plt.subplot(1, 2, 2)
    plt.imshow(I_right, cmap='gray')
    if len(matched_corners_right) > 0:
        plt.scatter(matched_corners_right[:, 1], matched_corners_right[:, 0], 
                    c='cyan', s=20, marker='+', label='Corners Detected Right')
    plt.title(f"Corners Detected Right ({len(matched_corners_right)} points)")
    plt.legend()
    plt.axis('off')

    # Display the visualization
    plt.tight_layout()
    plt.show()

    return np.array(depth_map)