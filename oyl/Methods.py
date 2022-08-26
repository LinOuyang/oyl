import numpy as np
import pandas as pd

def read_ascii(file,skip=1,dtype=float):
    f = open(file,'r')
    if skip>0:
        f.readlines(skip)
    data = []
    for line in f:
        d = [float(i) for i in line.split(' ') if i not in['','\n']]
        data.append(d)
    return np.array(data)


def maskout(x,y,points,groups=50):
    d = np.array(points)
    num = len(d)
    idx = np.argsort(d[:,0])
    d = d[idx,:]
    
    mask_shape = len(y),len(x)
    mask = np.zeros(mask_shape)
    size = np.round(num/groups)
    groups = np.int(np.ceil(num/size))
    x0 = np.zeros([groups])
    y0 = np.zeros([2,groups])

    for i in range(groups):
        st, ed = int(i*size), int(i*size+size)
        x0[i] = np.mean(d[st:ed,0])
        tmp = np.sort(d[st:ed,1])
        y0[0,i], y0[1,i] = tmp[0], tmp[-1]

    extend_x0 = np.hstack([np.array([x0[0]-0.001]),x0,np.array([x0[-1]+0.001])])
    p=np.argmin(np.abs(x.reshape(-1,1)-extend_x0),axis=1)
    start, end = np.where(p>0)[0][0],np.where(p<groups+1)[0][-1]

    for i in range(start,end+1):
        st = np.where(y>y0[0,p[i]-1])[0][0]
        ed = np.where(y<y0[1,p[i]-1])[0][-1]
        mask[st:ed+1,i] = 1

    mask = mask==0
    return mask

class Eof:
    
    def __init__(self,datasets,center=True):
        
        """
        data : an array with (n,***) shape. n is the time series grids
        return : an object with EOF analyzing methods
        """

        d = datasets.copy()
        n_times, *self._shape = d.shape
        d.shape = n_times, -1
        self._idx = np.isnan(d).any(axis=0)
        d = d[:,~self._idx]
        self.neofs = np.sum(~self._idx)
        if center:
            d = d-np.mean(d,0)
        cov = np.matmul(d.T,d)
        self._eig_val, self._eig_matrix = np.linalg.eig(cov)
        
        self._pcs = np.matmul(self._eig_matrix, d.T)

    def eofs(self,neofs=3):
        tmp = np.nan*np.zeros([neofs,np.prod(self._shape)])
        tmp[:,~self._idx] = self._eig_matrix[:,:neofs].T
        return tmp.reshape(neofs,*self._shape)

    def pcs(self,npcs=3):
        return self._pcs[:npcs].T

    def eigenvalues(self,neigs=None):
        if neigs is not None:
            r = self._eig_val[:neigs]
        else:
            r = self._eig_val
        return r

    def varianceFraction(self,neigs=None):
        return self.eigenvalues(neigs)/np.sum(self._eig_val)

class MultivariateEof:

    def __init__(self,datasets,center=True):
        
        data, info = self._merge_fields(datasets)
        self._shapes = info['shapes']
        self._slicers = info['slicers']
        self.solver = Eof(data,center=center)
        self.neofs = self.solver.neofs

    def eofs(self,neofs=3):
        modes = self.solver.eofs(neofs=neofs)
        return self._unwrap(modes)
    
    def pcs(self,npcs=3):
        return self.solver.pcs(npcs=npcs)

    def eigenvalues(self,neigs=None):
        return self.solver.eigenvalues(neigs)

    def varianceFraction(self,neigs=None):
        return self.solver.varianceFraction(neigs)


    def _unwrap(self, modes):
        """Split a returned mode field into component parts."""
        nmodes = modes.shape[0]
        modeset = [modes[:, slicer].reshape((nmodes,) + shape)
                   for slicer, shape in zip(self._slicers, self._shapes)]
        return modeset


    def _merge_fields(self, fields):
        """Merge multiple fields into one field.

        Flattens each field to (time, space) dimensionality and
        concatenates to form one field. Returns the merged array
        and a dictionary {'shapes': [], 'slicers': []} where the entry
        'shapes' is a list of the input array shapes minus the time
        dimension ans the entry 'slicers' is a list of `slice` objects
        that can be used to select each individual field from the merged
        array.

        """
        info = {'shapes': [], 'slicers': []}
        islice = 0
        for field in fields:
            info['shapes'].append(field.shape[1:])
            channels = np.prod(field.shape[1:])
            info['slicers'].append(slice(islice, islice + channels))
            islice += channels
        try:
            merged = np.concatenate(
                [field.reshape([field.shape[0], np.prod(field.shape[1:])])
                 for field in fields], axis=1)
        except ValueError:
            raise ValueError('all fields must have the same first dimension')
        return merged, info

def _get_shape(data,origin_scale,new_scale):
    origin_shape = data.shape
    origin_x_right = (origin_shape[1]-1)*origin_scale[0]
    origin_y_down = (origin_shape[0]-1)*origin_scale[1]
    s2, s1 = round(origin_x_right/new_scale[0])+1, round(origin_y_down/new_scale[1])+1
    s2 = s2+1 if s2*new_scale[0]<origin_x_right else s2
    s1 = s1+1 if s2*new_scale[1]<origin_y_down else s1
    return origin_shape,(s1,s2)

def inter(orgin,ref,n,shape):
    new_data = np.zeros(n)
    for i in range(n):
        index = int(i//ref)
        new_index = index if index == shape-1 else index+1
        delta = (i%ref)*(orgin[new_index]-orgin[new_index-1])/ref
        new_data[i] = orgin[index] + delta
    return new_data

def downscale(data,origin_scale,new_scale):
    origin_shape, new_shape = _get_shape(data,origin_scale,new_scale)
    new_data = np.zeros(new_shape)
    ref, conti = np.array(origin_scale)/np.array(new_scale), 0
    for j in range(0,new_shape[0]):
        index_origin = int(j//ref[1])
        if (conti==1)|(j%ref[1]==0):
            new_data[j, :], index_new = inter(data[index_origin, :], ref[0], new_shape[1], origin_shape[1]), j
            if index_origin+1<origin_shape[0]:
                gradi = inter(data[index_origin + 1, :], ref[0], new_shape[1], origin_shape[1]) - new_data[j, :]
        if (j%ref[1]+1)<=ref[1]:
            cul = new_data[index_new,:] if conti==0 else inter(data[index_origin, :], ref[0], new_shape[1], origin_shape[1])
            new_data[j,:] = cul + (j%ref[1])*gradi/ref[1]
            conti = 0
        elif (j%ref[1]+1)>ref[1]:
            conti = 1
            new_data[j, :] = new_data[index_new,:] + (j % ref[1]) * gradi/ref[1]
    return new_data

    


if __name__  == '__main__':
    l = np.zeros([80,82])
    print(downscale(l,[0.1,0.1],[0.05,0.05]).shape)
