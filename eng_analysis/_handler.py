"""
Contains the handler function to be used by all submodules only.
"""
import numpy as np


def handle_arrays(*args):
    """
    Makes sure that copies and numpy arrays with np.float64 datatypes are being used.

    :param args: Numpy arrays or lists to be handled
    :return: A list of copies of numpy arrays in args with np.float64 datatype
    """
    if len(args) > 1:
        return [np.array(item_, dtype=np.float64, copy=True) for item_ in args]
    else:
        return np.array(args, dtype=np.float64, copy=True)[0]
