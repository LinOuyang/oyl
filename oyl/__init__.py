import sys
import os
import pickle as pk
import datetime as dt
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import xarray as xr

from .Methods import *
from .Drawings import *
from .Scores import *
from . import utils


bar_color1 = 'darkturquoise'

def love():
    """
    A simple demo.
    """
    x = np.hstack([np.linspace(-1,-0.99,100),np.linspace(-0.99,0.99,800),np.linspace(0.99,1,100)])

    y = np.sqrt(1-x**2)+np.abs(x)
    plt.plot(x,y,color='orange',marker='.')

    y = -np.sqrt(1-x**2)+np.abs(x)
    plt.plot(x,y,'b.-')
    plt.legend([r'$y=\sqrt{1-x^2}+\left|x\right|$',r'$y=-\sqrt{1-x^2}+\left|x\right|$'])
    plt.show()




if __name__ == "__main__":
    arr = np.zeros([300, 20, 50])
    d = decompose_dim(arr, 0, [-1, 5, 2])
    print(d.shape)
    d = combine_dim(d, [0,1])
    print(d.shape)
    
