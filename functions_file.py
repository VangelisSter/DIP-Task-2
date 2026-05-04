import numpy as np

def myLocalDescriptor(I: np.ndarray, p: tuple, rhom: float, 
                      rhoM: float, rhostep: float, N: int) -> np.ndarray:
    M, K = I.shape
    # K rows, M columns

    p1, p2 = p[0], p[1]

    p_array = np.array(p)

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


        x_next = np.clip(x_base + 1, 0, K - 1)
        x_before = np.clip(x_base - 1, 0, K - 1)
        y_next = np.clip(y_base + 1, 0, M - 1)
        y_before = np.clip(y_base - 1, 0, M - 1)

        # Get the actual pixel intensity values from the image I
        top =    I[y_before, x_base]       # Top-Left
        bottom = I[y_next, x_base]       # Bottom-Left
        left =   I[y_base, x_before]       # Top-Right
        right =  I[y_base, x_next]       # Bottom-Right
    
        neigh_average = (top + bottom + left + right) / 4.0

        descriptor.append(np.mean(neigh_average))
    
    return np.array(descriptor)