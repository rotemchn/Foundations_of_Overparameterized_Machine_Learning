import numpy as np
from scipy.fft import dct
from numpy.linalg import matrix_rank
import matplotlib.pyplot as plt

# parameters
d = 128 # the dimensions of the input and output
n = 32 # number of samples
sigma_epsilon = 0.5  # it is given that sigma_epsilon^2= 0.25


C = dct(np.eye(d), axis=0, norm='ortho')

# make sure C is an orthonormal matrix
if not np.allclose(C.T @ C, np.eye(d)):
    raise Exception("C needs to be orthonormal!")


lambdas = 3 * 0.9 ** np.arange(d)
Lambda_x = np.diag(lambdas) # compute the diagonal matrix
Sigma_x = C @ Lambda_x @ C.T # compute the covariance matrix

beta = np.concatenate([np.full(20, 0.2178), np.full(d - 20, 0.0218)]) # beta~
beta = C @ beta # beta = Cbeta~

X = np.random.multivariate_normal(np.zeros(d), Sigma_x, n)  # get the x_train
epsilon = np.random.normal(0, sigma_epsilon, n) # random epsilon in range
y = X @ beta + epsilon # calculate y_train

# compute the rank of the matrix X
rank_X = matrix_rank(X)
print("Rank of X:", rank_X)

# create test data
n_test = 1000
X_test = np.random.multivariate_normal(np.zeros(d), Sigma_x, n_test) # random x_test matrix
epsilon_test = np.random.normal(0, sigma_epsilon, n_test) # random epsilon_test vector
y_test = X_test @ beta + epsilon_test

# learn the mode use Ridge Regression

test_errors_ls = []
gammas = [0.001, 0.1, 1.0]  # different gamma values for ridge regression
test_errors_r = [[] for _ in gammas]


for p in range(1, d + 1): # p = 1, 2, ..., d
    U_p = C[:, :p]
    Phi = X @ U_p

    # least squares solution
    if p > n:
        alpha_p = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    else:
        alpha_p = np.linalg.pinv(Phi) @ y
    predictions = alpha_p.T @ U_p.T @ X_test.T # calculate f_p(x)
    test_error = np.mean((predictions - y_test) ** 2) # average test error
    test_errors_ls.append(test_error)

    # ridge regression solution
    for i in range(len(gammas)):
        alpha_ridge = np.linalg.pinv(Phi.T @ Phi + gammas[i] * np.eye(p)) @ Phi.T @ y
        predictions_ridge = alpha_ridge.T @ U_p.T @ X_test.T
        test_error_ridge = np.mean((predictions_ridge - y_test) ** 2)
        test_errors_r[i].append(test_error_ridge)

# plot the empirical test error as a function of p
plt.figure(figsize=(10, 6))
plt.plot(range(1, d + 1), test_errors_ls, linewidth=3, color='cyan', label='Least Squares')
for i, g in enumerate(gammas):
    plt.plot(range(1, d + 1), test_errors_r[i], linestyle='--', label=f'Ridge Regression (gamma={g})')
plt.xlabel('p')
plt.ylabel('Test Error')
plt.title('Test Error as a function of Number of Parameters for Least Squares and Ridge Regression')
plt.grid(True)
plt.ylim(0, 2)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
plt.tight_layout()
plt.show()
