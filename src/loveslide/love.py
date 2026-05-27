import numpy as np
import os
from rpy2 import robjects
from rpy2.robjects import numpy2ri, packages, r


def call_love(X, lbd=0.5, mu=0.5, est_non_pure_row="HT", thresh_fdr=0.2, verbose=False, 
                    pure_homo=False, diagonal=False, delta=None, merge=False, 
                    rep_CV=50, ndelta=50, q=2, exact=False, max_pure=None, nfolds=10, outpath='.'):
    """
    Python wrapper for calling the LOVE function from local R scripts.
    Need to use local due to edge case where only 1 pure variable is found in LOVE.
    Sigma is also correlation (like in SLIDE) rather than covariance (like in LOVE)
    
    Returns:
    --------
    dict
        Dictionary containing the results from the LOVE analysis
    """

    gene_names = X.columns
    sample_names = X.index

    X = X.values
    
    # Activate automatic conversion between R and NumPy objects
    numpy2ri.activate()

    # Convert Python parameters to R values
    r_X = numpy2ri.py2rpy(X)
    r_lbd = robjects.FloatVector([lbd]) if not isinstance(lbd, (list, np.ndarray)) else robjects.FloatVector(lbd)
    r_mu = robjects.FloatVector([mu])
    r_est_non_pure_row = robjects.StrVector([est_non_pure_row])
    r_thresh_fdr = robjects.FloatVector([thresh_fdr])
    r_verbose = robjects.BoolVector([verbose])
    r_pure_homo = robjects.BoolVector([pure_homo])
    r_diagonal = robjects.BoolVector([diagonal])
    r_merge = robjects.BoolVector([merge])
    r_rep_CV = robjects.IntVector([rep_CV])
    r_ndelta = robjects.IntVector([ndelta])
    r_q = robjects.IntVector([q])
    r_exact = robjects.BoolVector([exact])
    r_nfolds = robjects.IntVector([nfolds])
    r_gene_names = robjects.StrVector(gene_names)
    r_sample_names = robjects.StrVector(sample_names)
    
    # Handle delta which can be None
    r_delta = robjects.NULL if delta is None else robjects.FloatVector([delta]) if not isinstance(delta, (list, np.ndarray)) else robjects.FloatVector(delta)
    
    # Handle max_pure which can be None
    r_max_pure = robjects.NULL if max_pure is None else robjects.IntVector([max_pure])


    # Source the correct LOVE function
    if pure_homo is True:
        love_function = 'getLatentFactors.R'
        love_directory = 'LOVE-SLIDE'
    else:
        love_function = 'LOVE.R'
        love_directory = 'LOVE-master/R'

    file_dir = os.path.dirname(os.path.abspath(__file__))
    love_path = os.path.join(file_dir, f"{love_directory}")
    
    # Source all R files in the directory - this will define the LOVE function
    r_files = [os.path.join(love_path, f) for f in os.listdir(love_path) 
               if f.endswith('.R')]
    
    love_main_file = None
    for file in r_files:
        if love_function in file:
            love_main_file = file
            break

    # If we found a main file, source it first
    if love_main_file:
        r(f'source("{love_main_file}")')
        r_files.remove(love_main_file)
    
    # Then source all other R files
    for r_file in r_files:
        r(f'source("{r_file}")')
    
    # Check if LOVE function is defined
    try:
        r("exists('LOVE')")
    except:
        raise ValueError("LOVE function not found. Please check the LOVE-master directory path.")
    
    # Now call the LOVE function directly using r.LOVE instead of r['LOVE']
    if pure_homo is True:
        result = r.getLatentFactors(r_X, 
                  delta=r_delta,
                  lbd=r_lbd, 
                  thresh_fdr=r_thresh_fdr,
                  rep_cv=r_rep_CV,
                  out_path=robjects.NULL, # Skip R plots to avoid font crashes on headless servers
                  verbose=r_verbose
                  )
    else:
        result = r.LOVE(r_X, 
                  delta=r_delta, 
                  lbd=r_lbd, 
                  mu=r_mu, 
                  thresh_fdr=r_thresh_fdr,
                  verbose=r_verbose,
                  pure_homo=r_pure_homo, 
                  est_non_pure_row=r_est_non_pure_row,
                  diagonal=r_diagonal,
                  merge=r_merge, 
                  rep_CV=r_rep_CV,
                  ndelta=r_ndelta, 
                  q=r_q, 
                  exact=r_exact, 
                  max_pure=r_max_pure, 
                  nfolds=r_nfolds,
                  out_path=outpath,
                  gene_names=r_gene_names,
                  sample_names=r_sample_names
                )
    
    # Convert R results to Python
    python_result = {}

    for i, key in enumerate(result.names):
        value = result.rx2(i + 1)  # 1-based indexing (R style)

        if key == "K":
            python_result[key] = int(value[0])  # single integer

        elif key in ["pureVec", "Gamma"]:
            python_result[key] = np.array(value)

        elif key in ["pureInd", "group"]:
            parsed = []
            for item in value:
                pos = list(map(int, item.rx2("pos"))) if "pos" in item.names else []
                neg = list(map(int, item.rx2("neg"))) if "neg" in item.names else []
                parsed.append({"pos": pos, "neg": neg})
            python_result[key] = parsed

        elif key in ["A", "C", "Omega"]:
            python_result[key] = np.array(value)

        elif key == "optDelta":
            python_result[key] = float(value[0])  # single float

        else:
            # Fallback: try conversion or store raw
            try:
                python_result[key] = np.array(value)
            except Exception:
                python_result[key] = value

    # Deactivate automatic conversion
    numpy2ri.deactivate()
    
    return python_result