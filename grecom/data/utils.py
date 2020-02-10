import numpy as np
import h5py
import scipy.sparse as sp


def load_matlab_file(path_file, name_field):
    """Load '.mat' files. Warning: '.mat' files should be saved in the
    '-v7.3' format

    :param path_file: File path
    :type path_file: str
    :param name_field: Field name
    :type name_field: str
    :return: Sparse float32 CSC matrix containing the data
    :rtype: scipy.sparse.csc_matrix
    """
    with h5py.File(path_file, 'r') as db:
        ds = db[name_field]
        try:
            if 'ir' in ds.keys():
                data = np.asarray(ds['data'])
                ir = np.asarray(ds['ir'])
                jc = np.asarray(ds['jc'])
                out = sp.csc_matrix((data, ir, jc)).astype(np.float32)
        except AttributeError:
            # Transpose in case is a dense matrix because of
            # the row- vs column- major ordering between python and matlab
            out = np.asarray(ds).astype(np.float32).T
    return out


def get_reltime(x):
    """Map array of absolute time to interval [0,1]

    :param x: Array with absolute timestamps
    :type x: numpy.array
    :return: Array with relative timestamps
    :rtype: numpy.array
    """
    return (x - x.min()) / (x.max() - x.min())


def get_logtime(x, scale=3600*24):
    """Map array of absolute time to logaritmic scale [0, inf)

    :param x: Array with absolute timestamps
    :type x: numpy.array
    :param scale: Time scale for mapping (seconds), defaults to 1 day
    :type scale: int, optional
    :return: Array with logaritmic timestamps
    :rtype: numpy.array"""
    return np.log(1 + (x - x.min())/scale)
