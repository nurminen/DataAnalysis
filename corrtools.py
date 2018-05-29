import numpy as np


def boot_corr_error(dada, nboot=int(10e3), CI=0.68):
    '''
    Computes bootstrapped (over trials) errors for correlations.
    dada:
    n neurons x n conditions x n trial array of spike counts
    numpy array of spike counts 

    OUTPUT:
    errs:n neurons x n conditions of bootstrapped CI


    '''

    
def euc_dist(v1, v2):
    '''
    euc_dist(v1, v2)

    Computes euclidean distance between two vectors. Throws an error if the vector are not of the same dimension

    '''
    assert (v1.shape[0] == v2.shape[0]), "Input vectors to euc_dist must have the same dimensions"
    return np.linalg(v1-v2, 2)
    
def boot_corr_Edistance(dada, dada_L, nboot=int(10e3)):
    '''
    boot_corr_distance(dada, nboot=int(10e3))
    
    Computes bootstrapped (over trials) correlations and uses the bootstrapped correlations to compute null distribution for 
    euclidean distance between the correlation functions with and without optogenetic silencing.
    
    dada:
    n neurons x n conditions x n trial array of spike counts
    numpy array of spike counts 
    
    dada_L:
    same as dada

    nboot:
    number of bootstrap samples to generate
    '''
    assert ((dada.shape[0] == dada_L.shape[0]) and (dada.shape[1] == dada_L.shape[1])), "0th and 1st dimensions of the input arrays to boot_corr_Edistance must be the same"
    
    # output matrix
    corrs_out   = np.nan*np.ones(((dada.shape[0]**2 - dada.shape[0])/2, dada.shape[1],nboot),float)
    corrs_out_L = np.nan*np.ones(((dada_L.shape[0]**2 - dada_L.shape[0])/2, dada_L.shape[1],nboot),float)
    
    # bootstrap indices
    boot_inds   = np.random.choice(dada.shape[2],(nboot, dada.shape[2]))
    boot_inds_L = np.random.choice(dada.shape[2],(nboot, dada.shape[2]))
    for diam in range(dada.shape[1]):
        print("Now booting diameter ",diam)
        for bi in range(boot_inds.shape[0]):
            C   = np.corrcoef(np.squeeze(dada[:,diam,boot_inds[bi,:]]))
            C_L = np.corrcoef(np.squeeze(dada[:,diam,boot_inds_L[bi,:]]))
            if diam == 0 and bi == 0:
                B = np.tril(np.ones(C.shape),-1) == 1

            # 
            corrs_out[:,diam,bi]   = C[B]
            corrs_out_L[:,diam,bi] = C_L[B]

    # boot norm
    b_norm = np.linalg.norm(corrs_out - corrs_out_L, ord=2, axis=1)
    return b_norm
