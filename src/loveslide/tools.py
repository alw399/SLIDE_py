from easydict import EasyDict
from collections import defaultdict
import os
import numpy as np
import pandas as pd

def init_data(input_params, x=None, y=None):
    """
    Initialize the data object and set default parameters.

    Args:
        input_params (dict): Dictionary containing input parameters.

    Returns:
        data (EasyDict): Data object with initialized parameters.
        input_params (dict): Dictionary containing input parameters with default values.
    """

    if 'n_workers' in input_params:
        raise ValueError('n_workers is no longer supported; knockoffs run serially. Remove it from input_params.')

    data = EasyDict()
    input_params = defaultdict(lambda: None, input_params)

    if input_params['x_path'] is None and x is None:
        raise ValueError("x_path is not provided")
    
    if input_params['y_path'] is None and y is None:
        raise ValueError("y_path is not provided")
    
    if input_params['y_factor'] is None:
        input_params['y_factor'] = True
    
    if input_params['delta'] is None:
        input_params['delta'] = [0.05, 0.1]

    if input_params['lambda'] is None:
        input_params['lambda'] = [0.1]

    if input_params['fdr'] is None:
        input_params['fdr'] = 0.1
    
    if input_params['thresh_fdr'] is None:
        input_params['thresh_fdr'] = 0.2

    if input_params['pure_homo'] is None:
        input_params['pure_homo'] = True

    if input_params['spec'] is None:
        input_params['spec'] = 0.2

    if input_params['niter'] is None:
        input_params['niter'] = 100
    
    if input_params['SLIDE_top_feats'] is None:
        input_params['SLIDE_top_feats'] = 20

    if input_params['out_path'] is None:
        input_params['out_path'] = os.getcwd()
    
    if input_params['do_interacts'] is None:
        input_params['do_interacts'] = True

    if x is None:
        data.X = pd.read_csv(input_params['x_path'], index_col=0)
    else:
        data.X = x
    
    if y is None:
        data.Y = pd.read_csv(input_params['y_path'], index_col=0)
    else:
        data.Y = y

    if input_params['y_factor'] is True:
        data.Y = data.Y.replace({
            orig_y: i for i, orig_y in enumerate(np.unique(data.Y))
        })
        data.Y = data.Y.astype(int)

    return data, input_params

def show_params(input_params, data):
    """
    Display the parameters and data.

    Args:
        input_params (dict): Dictionary containing input parameters.
        data (EasyDict): Data object with initialized parameters.
    """
    print(f'\n### PARAMETERS ###\n')
    for k, v in input_params.items():
        print(f"{k}: {v}")
    
    print(f'\n###### DATA ######\n')
    print(f'{data.Y.shape[0]} samples')
    print(f'{data.X.shape[1]} features')
    print(f'{(data.Y == 1).values.sum() / len(data.Y) * 100:.1f}% cases')
    print(f'{(data.Y == 0).values.sum() / len(data.Y) * 100:.1f}% controls')
    
    check_params(input_params, data)
    print(f'\n##################\n')

def check_params(input_params, data):
    """
    Check the parameters and data.

    Args:
        input_params (dict): Dictionary containing input parameters.
        data (EasyDict): Data object with initialized parameters.
    """

    zero_std_cols = data.X.columns[data.X.std() == 0]
    if len(zero_std_cols) > 0:
        print(f"Warning: Found {len(zero_std_cols)} features with 0 standard deviation. These will be removed.")
        data.X = data.X.loc[:, data.X.std() > 0]


def calc_default_fsize(n_rows, K):
    """
    Calculate the default f_size.

    Parameters:
    - n_rows: integer representing the number of samples
    - K: integer representing the number of latent factors

    Returns:
    - Integer representing the default f_size
    """
    
    # written exactly as in the R code
    f_size = K 
    
    if (n_rows <= K) and (K < 100):
        if abs(n_rows - K) <= 2:
            f_size = n_rows - 2
        else:
            f_size = n_rows
            
    if (n_rows > K) and (K < 100):
        f_size = K
        
    if n_rows < K:
        f_size = n_rows
        
    return f_size
