"""
Data generators and related stuff
"""

import os
import scipy.io
import numpy as np
import imageio
from PIL import Image, ImageOps

def get_filenames(path, extension):
    return [x for x in os.listdir(path) if x.endswith(extension)]

def data_generator_BSDS(x_path, y_path):
    """ 
    generate one pair of (x,y) for each label y of each sample x 
    Note: the BSDS datasets have more than 1 label image for each sample
          using each x-y pair for each sample as batches
    """

    x_path += '/' if not x_path.endswith('/') else ''
    y_path += '/' if not y_path.endswith('/') else ''
    
    sample_names = [n[:-4] for n in get_filenames(x_path, '.jpg')]

    in_shape = np.array(Image.open("{}{}.jpg".format(x_path,sample_names[0]))).shape
    if (in_shape[0]%2 != 0 or in_shape[1]%2 != 0):
        in_shape = (in_shape[0] - in_shape[0]%2, in_shape[1] - in_shape[1]%2, in_shape[2])

    for s in sample_names:
        
        x = np.array(Image.open("{}{}.jpg".format(x_path,s)).resize((in_shape[1], in_shape[0])))
        if x.shape != in_shape: x = x.transpose([1,0,2])
        
        y_all = scipy.io.loadmat("{}{}.mat".format(y_path, s))

        x_arr = []
        y_arr = []
        for j in range(y_all['groundTruth'].shape[1]):
            y = np.array(Image.fromarray(y_all['groundTruth'][0,j][0][0][1]).resize((in_shape[1], in_shape[0])))
            if y.shape != in_shape[:-1]: y = y.T
            x_arr += [x/255.0]
            y_arr += [y.astype(float)]

        yield(np.array(x_arr), np.array(y_arr))
        
def data_generator_pathfinder(data_root, batch_size=8):
    """
    data generator for the BSDS dataset
    - Note: y-encoding is [P(negative), P(positive)]
    """
    
    data_root += '/' if not data_root.endswith('/') else ''

    x_files = []
    y_arr_all = []
    
    # positive samples
    x_path = data_root + "curv_baseline/imgs/"
    dirs = [d for d in os.listdir(x_path) if os.path.isdir(os.path.join(x_path,d))]
    for d in dirs:
        for f in get_filenames(x_path+d, '.png'):
            x_files += ["{}{}/{}".format(x_path,d,f)]
            y_arr_all += [[0,1]]

    # negative samples
    x_path = data_root + "curv_baseline_neg/imgs/"
    dirs = [d for d in os.listdir(x_path) if os.path.isdir(os.path.join(x_path,d))]
    for d in dirs:
        for f in get_filenames(x_path+d, '.png'):
            x_files += ["{}{}/{}".format(x_path,d,f)]
            y_arr_all += [[1,0]]

    for i in range(len(x_files) // batch_size):
        # TODO: not handling the last batch
        x_arr = np.empty((0,300,300,1))
        y_arr = np.empty((0,2))
        for j in range(batch_size):
            x_arr = np.vstack((x_arr, imageio.imread(x_files[i*batch_size+j]).reshape(1,300,300,1)))
            y_arr = np.vstack((y_arr, np.array(y_arr_all[i*batch_size+j])))
        yield (x_arr, y_arr)