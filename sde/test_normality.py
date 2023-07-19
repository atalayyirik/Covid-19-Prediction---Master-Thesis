import numpy as np
import scipy.spatial


def test_normality(batch_samples, test_standard_normal=True):
    """
    test_normality of the given samples.
    We only test w.r.t. the standard normal, not for "any" normal distribution.
    
    Based on: https://doi.org/10.1007/BF02613322
    
    Parameters
    ----------
    batch_samples
        N x d matrix of N points in d dimensions.
    test_standard_normal
        if True, will test if the batch is a standard normal N(0,1), otherwise
        will test for arbitrary normal distribution.
    Returns
    -------
    Test statistic Tn from the paper.
    """
    
    # parameters
    n = batch_samples.shape[0]
    d = batch_samples.shape[1]
    
    # sample mean and covariance
    if test_standard_normal:
        # these are not used, but implicitly that is what we use
        Sinv = np.diag(np.ones((d,)))
        mean = np.zeros((1,d))
        
        # squared pairwise distances with covariance = I
        Rjk = scipy.spatial.distance.pdist(batch_samples, 'sqeuclidean')
        Rjk = scipy.spatial.distance.squareform(Rjk)
        
        # squared distances to mean zero
        Rj2 = np.linalg.norm(batch_samples-mean, axis=1)**2
    else:
        if d > 1:
            Sinv = np.linalg.pinv(np.cov(batch_samples.T))
        else:
            Sinv = np.array([1/np.cov(batch_samples.T)]).reshape(1,1)
        mean = np.mean(batch_samples).reshape(1,-1)
    
        # squared pairwise distances
        Rjk = scipy.spatial.distance.pdist(batch_samples, 'mahalanobis', VI=Sinv)
        Rjk = scipy.spatial.distance.squareform(Rjk**2)
    
        # squared distances to mean zero
        Rj2 = np.array([
            (batch_samples[k,:]-mean) @ Sinv @ (batch_samples[k,:]-mean).T
            for k in range(n)
        ])
    
    # test statistic
    Tn0 = 1/n * np.sum(np.exp(-0.5*Rjk))
    Tn1 = -2**(1-d/2) * np.sum(np.exp(-0.25 * Rj2))
    Tn = Tn0 + Tn1 + n*3**(-d/2)
    
    return Tn