import numpy as np

def myLocalDescriptor(I: np.ndarray, p: tuple, rhom: float, 
                      rhoM: float, rhostep: float, N: int) -> np.ndarray:
    K, M = I.shape
    # K rows, M columns

    p_array = np.flip(np.array(p)) # To transform from row,col to x,y

    # Out of Bounds check
    if (p_array > [M, K] or p_array < rhoM).any():
        return np.empty(0)
    
    #rho = [i for i in range(rhom, rhoM, rhostep)]
    radii = np.arange(rhom, rhoM + rhostep, rhostep)
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    descriptor = []

    for r in radii:
        # x = p1 + r * cos(theta)
        [x_exact, y_exact] = p_array + r * [cos_theta, sin_theta]
        
        # We round down to get the exact pixel
        x_base = np.floor(x_exact).astype(int)
        y_base = np.floor(y_exact).astype(int)


        # Find neighbouring pixels while minding the boundaries
        x_next = np.clip(x_base + 1, 0, M - 1)
        x_before = np.clip(x_base - 1, 0, M - 1)
        y_next = np.clip(y_base + 1, 0, K - 1)
        y_before = np.clip(y_base - 1, 0, K - 1)

        # Get the actual pixel intensity values from the image I
        I_top =    I[y_before, x_base]       # Top
        I_bottom = I[y_next, x_base]       # Bottom
        I_left =   I[y_base, x_before]       # Left
        I_right =  I[y_base, x_next]       # Right
    
        # First find the mean of the neighbours
        neigh_mean = (I_top + I_bottom + I_left + I_right) / 4.0

        # Then find the mean of the N values regarding this radius
        descriptor.append(np.mean(neigh_mean))
    
    return np.array(descriptor)

def isCorner(I: np.ndarray, p: tuple, k: float, Rthres: float) -> bool:

    #Exctract the points and dimensions
    p1, p2 = p
    K, N = I.shape
    # K rows, N columns

    # We make the assumption the weight uniform, as in in the neighbour of p it is 1 and elsewere 0

    # First we need the neighbouring pixels
    # We use +2 for the ends because Python slicing is exclusive at the upper bound
    r_start = max(0, p1 - 1)
    r_end   = min(K, p1 + 2) 
    c_start = max(0, p2 - 1)
    c_end   = min(N, p2 + 2)

    # 2. Extract the 3x3 patch
    patch = I[r_start:r_end, c_start:c_end]


    # Calculate the derivative arrays
    Iy, Ix = np.gradient(patch)

    # Calculate the M matrix components
    M11 = np.sum(Ix ** 2)
    M12 = np.sum(Ix * Iy)
    M22 = np.sum(Iy ** 2)

    # Calculate the determinent and trace

    det_M = (M11 * M22) - (M12 ** 2)
    trace_M = M11 + M22

    # Calculate R
    R = det_M - k * (trace_M ** 2)

    return R > Rthres

def myDetectHarrisFeatures(I: np.ndarray) -> np.ndarray:
    corners = []
    K, N = I.shape
    
    # Define parameters 
    k = 0.04
    Rthres = 0.01
    
    # 2. Loop over the pixels
    # We skip the outermost 1-pixel border to ensure the 3x3 window 
    # of the isCorner function fits perfectly inside the image without edge distortion.
    for row in range(1, K - 1):
        for col in range(1, N - 1):
            
            p = (row, col)
            
            # 3. Call the function as instructed
            if isCorner(I, p, k, Rthres):
                # The text asks for a 2-column array where each row is the coordinates
                corners.append([row, col])
                
    # 4. Convert the list to a NumPy array before returning
    return np.array(corners)

def computeDisparity(p1: tuple, p2: tuple) -> float:
    D = 144.049 #mm
    f = 4396.869 #px
    doffs = 185.788 #px
