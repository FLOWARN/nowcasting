import numpy as np
from scipy import signal
from sklearn.metrics.cluster import contingency_matrix
from numba import jit


def FSS(pred, gt, threshold=8., n=8):
    """General definition for Fraction Skill Score for any value of n and any threshold

    Args:
        pred (_type_): _description_
        gt (_type_): _description_
        threshold (_type_, optional): _description_. Defaults to 8..
        n (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    pred = pred.copy()
    gt = gt.copy()
    
    
    pred[pred<=threshold] = 0
    pred[pred>threshold] = 1
    gt[gt<=threshold] = 0
    gt[gt>threshold] = 1

    # pred = (pred > threshold)*np.ones_like(pred)

    convolution_matrix = np.ones((n,n))/(n^2)
    if n >=1:
        pred = signal.convolve2d(pred, convolution_matrix, mode='valid')
        gt = signal.convolve2d(gt, convolution_matrix, mode='valid')
    FSS = 2*(np.multiply(pred,gt).sum())/(np.square(pred).sum() + np.square(gt).sum())
    
    return FSS

def RMSE(pred, gt):
    return np.sqrt(MSE(pred, gt))

def MSE(pred, gt):
    return np.mean(np.square(pred-gt).ravel())

def R_squared(pred, gt):
    mean = np.mean(pred)
    residual_squared = np.square(pred-gt).sum()
    sum_of_squares = np.square(pred-mean).sum()
    R_squared = residual_squared/sum_of_squares
    return R_squared

    
def get_contingency_table(pred, gt, threshold):
    pred[pred<=threshold] = 0
    pred[pred>threshold] = 1
    gt[gt<=threshold] = 0
    gt[gt>threshold] = 1
    
    contingency_table = contingency_matrix(gt, pred)
    a = contingency_table[0][0]
    b = contingency_table[0][1]
    c = contingency_table[1][0]
    d = contingency_table[1][1]
    
    return contingency_table


def HeidkeSkillScore(pred, gt, threshold):
    """Calculate HSS as per (https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2014/Scores1.pdf)
    -inf < HSS <= 1
    HSS < 0 has no skill
    Args:
        pred (_type_): _description_
        gt (_type_): _description_
        threshold (_type_): _description_

    Returns:
        _type_: _description_
    """
    contingency_table = get_contingency_table(pred, gt, threshold)
    a = contingency_table[0][0]
    b = contingency_table[0][1]
    c = contingency_table[1][0]
    d = contingency_table[1][1]
    
    HSS = 2*(a*d - b*c)/((a+c)*(c+d)*(a+b)*(b+d))
    return HSS
