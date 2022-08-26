import numpy as np
import matplotlib.pyplot as plt
from .Methods import *
from .Scores import *

bar_color1 = 'darkturquoise'

def timing(aim = None):
    if aim is not None:
        now = dt.datetime.now()
        tmp = aim - now
        res = tmp.seconds
        print('waitting for {} seconds'.format(res))
        time.sleep(res)

def decompose_dim(arr, dim=0, *shape):
    """
    For input arrays, the specified dimensions are decomposed into the specified shapes
    
    Example:
    >>> arr = np.zeros([30, 200, 50])
    >>> d = decompose_dim(arr, 1, (20,-1))
    >>> d.shape
    (30, 20, 10, 50)
    """
    arr_shape = list(arr.shape)
    if np.iterable(shape[0]):
        shape = shape[0]
    if (-1 not in shape )&(np.prod(shape) != arr_shape[dim]):
        raise ValueError(f"The length of dimension {dim} must keep the same in decomposition")
    arr_shape[dim] = shape
    new_shape = []
    for i in arr_shape:
        if not np.iterable(i):
            new_shape.append(i)
        else:
            new_shape += list(i)
    return arr.reshape(new_shape)

def combine_dim(arr, *dim):
    """
    For the input array, reshape the specified dimension, making it a uniform dimension.
    
    Example:
    >>> arr = np.zeros([30, 10, 20, 50])
    >>> d = combine_dim(arr, 0, 1)
    >>> d.shape
    (300, 20, 50)
    """
    if np.iterable(dim[0]):
        dim = dim[0]
    else:
        dim = list(dim)
    shape = np.array(arr.shape)
    dim = np.sort(np.arange(len(shape))[dim])
    if ((dim[1:] - dim[:-1])!=1).any():
        warn_mes = "The input dimensions must be adjacent, otherwise other problems may result"
        np.warnings.warn(warn_mes, np.VisibleDeprecationWarning)
    shape[dim[-1]] = np.prod(shape[dim])
    shape = np.delete(shape, dim[:-1])
    return arr.reshape(shape)

def view(arr, **kwargs):
    """
    imshow an matrix
    """
    plt.colorbar(plt.imshow(arr, **kwargs)), plt.show()

def stat(arr, name='?', ndot=1):
    """
    print the data loss rate, mean variance,
    minimum, mean, maximum
    """
    nan_num = np.isnan(arr).sum()
    nan_rate = nan_num/np.prod(arr.shape)
    vmin, vmean, vmax = np.nanmin(arr), np.nanmean(arr), np.nanmax(arr)
    vmin, vmean, vmax = np.round(vmin, ndot), np.round(vmean, ndot), np.round(vmax, ndot)
    sd = np.sqrt(np.nanmean((arr-vmean)**2))
    sd = np.round(sd, ndot)
    print(f"{name}: {nan_rate*100:.1f}%, {arr.shape} {sd}\n{vmin}, {vmean}, {vmax}\n")

if __name__ == "__main__":
    arr = np.zeros([300, 20, 50])
    d = decompose_dim(arr, 0, [-1, 5, 2])
    print(d.shape)
    d = combine_dim(d, [0,1])
    print(d.shape)
    
