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

# compute the rank of the matrix X - question a
rank_X = matrix_rank(X)
print("Rank of X:", rank_X)

# create test data - question b
n_test = 1000
X_test = np.random.multivariate_normal(np.zeros(d), Sigma_x, n_test) # random x_test matrix
epsilon_test = np.random.normal(0, sigma_epsilon, n_test) # random epsilon_test vector
y_test = X_test @ beta + epsilon_test

# learn the mode - question c

test_errors = []
for p in range(1, d + 1): # p = 1, 2, ..., d
    U_p = C[:, :p]
    Phi = X @ U_p
    if p > n:
        alpha_p = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
    else:
        alpha_p = np.linalg.pinv(Phi) @ y
    predictions = alpha_p.T @ U_p.T @ X_test.T # calculate f_p(x)
    test_error = np.mean((predictions - y_test) ** 2) # average test error
    test_errors.append(test_error)

# plot the empirical test error as a function of p
plt.figure(figsize=(10, 6))
plt.plot(range(1, d + 1), test_errors, marker='o')
plt.xlabel('p')
plt.ylabel('Test Error')
plt.title('Test Error as a function of p = 1,...,d')
plt.grid(True)
plt.show()

# question d - do the experiment for K times
K = 500

test_errors = np.zeros((d, K)) # initialize errors as 0
f_avg = np.zeros((n_test, d))
f = np.zeros((n_test, d, K))

for k in range(K):
    # new training set for each k
    X = np.random.multivariate_normal(np.zeros(d), Sigma_x, n)
    epsilon = np.random.normal(0, sigma_epsilon, n)
    y = X @ beta + epsilon

    for p in range(1, d + 1): # like in section c, learn the model with the new training set and predict
        U_p = C[:, :p]
        Phi = X @ U_p
        if p > n:
            alpha_p = np.linalg.pinv(Phi.T @ Phi) @ Phi.T @ y
        else:
            alpha_p = np.linalg.pinv(Phi) @ y
        predictions = alpha_p.T @ U_p.T @ X_test.T  # calculate f_p(x)

        test_error = np.mean((predictions - y_test) ** 2)
        test_errors[p - 1, k] = test_error # add current test error to the right matrix place

        # store predictions for bias and variance calculation
        f[:, p - 1, k] = predictions

    # compute the average prediction over all K datasets for each p
    f_avg = np.mean(f, axis=2)

# compute the average test error over K datasets for each p
avg_test_errors = np.mean(test_errors, axis=1)

# compute the squared bias
squared_bias = np.mean((f_avg - X_test @ beta[:, None]) ** 2, axis=0)

# compute the variance
variance = np.mean((f - f_avg[:, :, None]) ** 2, axis=(0, 2))

# plot the results
plt.figure(figsize=(15, 8))
plt.plot(range(1, d + 1), avg_test_errors, label='Test Error')
plt.plot(range(1, d + 1), squared_bias, label='Bias Squared')
plt.plot(range(1, d + 1), variance, label='Variance')

plt.xlabel('p')
plt.ylabel('Error')
plt.title('Test Error, Bias Squared, and Variance as a function of p')
plt.legend()
plt.grid(True)
plt.ylim(0, 5) # limit the y-axis, so we can see what happens around 0 error
plt.show()