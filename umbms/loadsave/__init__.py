"""
Tyson Reimer
University of Manitoba
June 19th, 2019
"""

import pickle

###############################################################################


def load_pickle(path):
    """Loads a pickle file

    Parameters
    ----------
    path : str
        The full path to the .pickle file that will be loaded

    Returns
    -------
    loaded_var :
        The loaded variable, can be array_like, int, str, dict_to_save,
        etc.
    """

    with open(path, 'rb') as handle:
        loaded_var = pickle.load(handle)

    return loaded_var


def save_pickle(var, path):
    """Saves the var to a .pickle file at the path specified

    Parameters
    ----------
    var : object
        A variable that will be saved
    path : str
        The full path of the saved .pickle file
    """

    with open(path, 'wb') as handle:
        pickle.dump(var, handle, protocol=pickle.HIGHEST_PROTOCOL)
