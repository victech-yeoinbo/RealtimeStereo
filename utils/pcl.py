import numpy as np

def query_intrinsic(cat):
    if cat == 'withrobot':
        f = 467.83661057
        cx, cy = 284.1095847, 256.36649503
        baseline = 0.120601 # abs(P2[1,4]) / Q[3,4]
    elif cat == 'dexter':
        f = 320
        cx, cy = 0.5*640-0.5, 0.5*480-0.5
        baseline = 0.12
    else:
        raise RuntimeError('Not supported dataset category')

    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float32)
    return K, baseline


def populate_pc(depth, K, flatten=True):
    '''
    convert depth to point clouds of (x,y,z) world coordinate sysatem = right-handed 
        (Z+)
       /
      /
     /
    +------> (X+)
    |
    |
    |
    (Y+)
    '''
    H, W = depth.shape[:2]
    py, px = np.mgrid[:H,:W]
    xyz = np.stack([px, py, np.ones_like(px)], axis=-1) * np.atleast_3d(depth)
    if flatten:
        xyz = np.reshape(xyz, (-1, 3))
    return xyz @ np.linalg.inv(K).T

