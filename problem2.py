#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.integrate import nquad
from numpy.linalg import eigvals
# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html

# Set random seed for reproducibility
np.random.seed(42)



# Generate two multivariate Gaussian random variable objects
prior1, mean1, cov1 = 0.35, np.array([-1, -1, -1, -1]), np.array([[5, 3, 1, -1], [3, 5, -2, -2], [1, -2, 6, 3], [-1, -2, 3, 4]])
rv1 = multivariate_normal(mean1, cov1)

prior2, mean2, cov2 = 0.65, np.array([1, 1, 1, 1]), np.array([[1.6, -0.5, -1.5, -1.2], [-0.5, 8, 6, -1.7], [-1.5, 6, 6, 0], [-1.2, -1.7, 0, 1.8]])
rv2 = multivariate_normal(mean2, cov2)

total = 10000

data1 = rv1.rvs(size = int(total * prior1))
data2 = rv2.rvs(size = int(total * prior2))

def int_error(x1, x2, x3, x4, rv1, rv2, prior1, prior2):
    """
    Inside of error calculation integral
    """
    pos = [x1, x2, x3, x4]
    err_1 = rv1.pdf(pos) * prior1
    err_2 = rv2.pdf(pos) * prior2
    return min(err_1, err_2)

def classify(x, gamma, rv1, rv2):
    """
    Return if x belongs to L1, false for L0
    """
    p2 = rv2.pdf(x) 
    p1 = rv1.pdf(x)
    return (p2/p1) > gamma

def get_roc_values(gamma, rv1, rv2, b):
    """
    Return the FPR, TPR, and error for the provided random values
    """
    tp = 0
    fp = 0
    error = 0
    for l0 in data1:
        if classify(l0, gamma, rv1, rv2):
            fp += 1
            error += b

    for l1 in data2:
        if classify(l1, gamma, rv1, rv2):
            tp += 1
        else:
            error += 1

    return (fp / (total * prior1), tp / (total * prior2), error / total, gamma)

def calculate_roc_error(rv1, rv2, label):
    """
    Returns the FPR/TPR curve and error/gamma curve for the provided random values
    """
    fp = []
    tp = []
    error = []
    gs = []
    
    # Calculate error bound
    
    #err = nquad(int_error, ranges=[[-5, 5],[-5, 5],[-5, 5],[-5, 5]], args = [rv1, rv2, prior1, prior2], opts = {'epsabs': 1.e-2})
    #print(f"{label} bound on error is {err}")

    for g in np.logspace(-2, 5, 300):
        val = get_roc_values(g, rv1, rv2, 1)
        fp.append(val[0])
        tp.append(val[1])
        error.append(val[2])
        gs.append(g)
        
    smallest_err = np.argmin(error)
    smallest_g = gs[smallest_err]
    best_fp, best_tp, best_wrong, best_g = get_roc_values(smallest_g, rv1, rv2, 1)

    print(f"{label}: smallest gamma is {smallest_g} (FPR {best_fp}, TPR {best_tp}, Error % {best_wrong}) ")

    return (fp, tp, error, gs)

def calculate_err_with_risk(rv1, rv2):
    """
    Creates the gamma/B and error/B curve
    """
    error = []
    bs = []
    gs = []

    g = prior1 / prior2

    for b in np.arange(0, 100, 0.5):
        val = get_roc_values(b * g, rv1, rv2, b)
        error.append(val[2])
        gs.append(b * g)
        bs.append(b)

    return (bs, gs, error)
    

# Q2 Part A random variables
true_fp, true_tp, true_errors, true_gammas = calculate_roc_error(rv1, rv2, "True COV")

# Q2 Part C random variables
diag_cov1 = np.array([[5, 0, 0, 0], [0, 5, 0, 0], [0, 0, 6, 0], [0, 0, 0, 4]])
diag_rv1 = multivariate_normal(mean1, diag_cov1)

diag_cov2 = np.array([[1.6, 0, 0, 0], [0, 8, 0, 0], [0, 0, 6, 0], [0, 0, 0, 1.8]])
diag_rv2 = multivariate_normal(mean2, diag_cov2)

diag_fp, diag_tp, diag_errors, diag_gammas = calculate_roc_error(diag_rv1, diag_rv2, "Diag COV")

# Q2 Part D random variables
sum = np.empty((4, 4))
for x in data1:
    mean_diff = np.atleast_2d(x - mean1)
    sum += np.matmul(mean_diff, np.transpose(mean_diff))

for x in data2:
    mean_diff = np.atleast_2d(x - mean2)
    sum += np.matmul(np.transpose(mean_diff), mean_diff)

shared_cov = sum * (1/total)

shared_rv1 = multivariate_normal(mean1, shared_cov)
shared_rv2 = multivariate_normal(mean2, shared_cov)

shared_fp, shared_tp, shared_errors, shared_gammas = calculate_roc_error(shared_rv1, shared_rv2, "Shared COV")

# Q2 Part E
risk_betas, risk_errors, risk_gammas = calculate_err_with_risk(rv1, rv2)

# Q2 Part B 
bs = []
ch_errs = []

def kb(u1, u2, s1, s2, b):
    gamma_part = b * s1 + (1-b) * s2
    gamma_inv = np.linalg.inv(gamma_part)
    mean_diff = np.atleast_2d(u2 - u1)

    mat_part = np.matmul(np.matmul(mean_diff, gamma_inv), np.transpose(mean_diff))[0][0]

    p1 = ((b*(1-b))/2) * mat_part
    p2 = 0.5 * np.log(np.linalg.det(gamma_part) / ((np.linalg.det(s1) ** b) * (np.linalg.det(s2) ** (1-b))))
    return p1 + p2

def chernoff_err(b):
    return (prior1 ** b) * (prior2 ** (1-b)) * np.exp(-kb(mean1, mean2, cov1, cov2, b))

for b in np.arange(0, 1, 0.01):
    bs.append(b)
    ch_errs.append(chernoff_err(b))

smallest_err = np.argmin(ch_errs)
smallest_b = bs[smallest_err]

print(f"Smallest Chernoff beta is {smallest_b}, error of {ch_errs[smallest_err]}")

b_bound = chernoff_err(0.5)

print(f"Bhattacharyya bound is {b_bound}")


fig, axes = plt.subplots(nrows=5, ncols=2)
ax1 = axes[0][0]
ax1.scatter(true_fp, true_tp)
ax1.set_xlabel('FPR (True covariances)')
ax1.set_ylabel('TPR')
ax1.legend()

ax2 = axes[0][1]
ax2.scatter(true_gammas, true_errors)

ax2.set_xlabel('$\\gamma$ (True covariances)')
ax2.set_ylabel('Error %')
ax2.legend()

ax1 = axes[1][0]
ax1.scatter(diag_fp, diag_tp)
ax1.set_xlabel('FPR (Diagonal covariances)')
ax1.set_ylabel('TPR')
ax1.legend()

ax2 = axes[1][1]
ax2.scatter(diag_gammas, diag_errors)

ax2.set_xlabel('$\\gamma$ (Diagonal covariances)')
ax2.set_ylabel('Error %')
ax2.legend()

ax1 = axes[2][0]
ax1.scatter(diag_fp, diag_tp)
ax1.set_xlabel('FPR (Shared covariances)')
ax1.set_ylabel('TPR')
ax1.legend()

ax2 = axes[2][1]
ax2.scatter(diag_gammas, diag_errors)

ax2.set_xlabel('$\\gamma$ (Shared covariances)')
ax2.set_ylabel('Error %')
ax2.legend()

ax1 = axes[3][0]
ax1.scatter(risk_betas, risk_gammas)
ax1.set_xlabel('$\\beta$')
ax1.set_ylabel('$\\gamma$')
ax1.legend()

ax2 = axes[3][1]
ax2.scatter(risk_betas, risk_errors)

ax2.set_xlabel('$\\beta$')
ax2.set_ylabel('Risk %')
ax2.legend()

ax3 = axes[4][0]
ax3.scatter(bs, ch_errs)

ax3.set_xlabel('$\\beta$')
ax3.set_ylabel('Chernoff bound')
ax3.legend()

plt.savefig("two-gaussians.pdf", format="pdf", bbox_inches="tight")
plt.show()
