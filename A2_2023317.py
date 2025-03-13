import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

### Functions for preprocessing data

def load_data():
    """
    Loads training and testing datasets from mnist
    reduces data to only 0,1,2 and returns it.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #used for debugging
    '''
    print(type(x_train), x_train.shape, x_train.ndim, x_train.dtype, x_train.itemsize)

    with open('mnist_data.txt', 'w') as f:
        f.write("x_train1: \n" + str(x_train) + "\n")
        f.write("y_train: \n" + str(y_train) + "\n")
        f.write("x_test: \n" + str(x_test) + "\n")
        f.write("y_test: \n" + str(y_test) + "\n")

    print("xtr: ", len(x_train), "ytr: ", len(y_train), "xts: ", len(x_test), len(y_test))
    '''

    train_index = np.isin(y_train, [0, 1, 2])
    #output of above function: [False True ...]
    #print("ms",train_index)
    test_index = np.isin(y_test, [0, 1, 2])

    #reduced data set to only those with digits 0,1,2
    x_train, y_train = x_train[train_index], y_train[train_index]
    x_test, y_test = x_test[test_index], y_test[test_index]

    # Normalizing to range [0,1]
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0

    return x_train, y_train, x_test, y_test

def sample_data(x, y, sample_num=100):
    """
    choose sample_num number of samples randomly and return
    """
    sampled_x, sampled_y = [], []
    for label in [0, 1, 2]:
        indices = np.where(y == label)[0]
        #replace is false so no repetition and 100 samples randomly chosen
        sampled_indices = np.random.choice(indices, sample_num, replace=False)     
        sampled_x.append(x[sampled_indices])
        #sampled_x: [array([0,0,...],uint), array(), array()]
        sampled_y.append(y[sampled_indices])

    return np.vstack(sampled_x), np.concatenate(sampled_y)

def preprocess_data():
    """
    Loads dataset and filters values to 0,1,2
    Convert images to feature and normalize it. 
    """
    x_train, y_train, x_test, y_test = load_data()
    # sample_x, sample_y = sample_data(x_train, y_train, 200)
    # x_train = np.concatenate([sample_x[0:100], sample_x[200:300], sample_x[400:500]], axis=0)
    # y_train = np.concatenate([sample_y[0:100], sample_y[200:300], sample_y[400:500]], axis=0)

    # x_test = np.concatenate([sample_x[100:200], sample_x[300:400], sample_x[500:600]], axis=0)
    # y_test = np.concatenate([sample_y[100:200], sample_y[300:400], sample_y[500:600]], axis=0)
    x_train, y_train = sample_data(x_train, y_train)
    x_test, y_test = sample_data(x_test, y_test)
    return x_train, y_train, x_test, y_test


### Functions used for mle estimate

def covariance(x_data, x_mean, biased=0):
    x_data-=x_mean
    #print("x_data", (x_data).shape)
    return (1/(len(x_data)-biased))*((x_data.T)@(x_data))

def mean_and_cov(x_train, y_train, biased=0):
    means, covariances = {}, {}

    for c in [0,1,2]:
        x_c = x_train[y_train == c]
        means[c] = np.mean(x_c, axis=0)
        covariances[c] = covariance(x_c, means[c], biased)
        #print("cov: ",covariances[c], len(covariances[c]))

    return means, covariances

### Function used for PCA

def pca(X, minimum=0.95, num_com=False):
    mean = np.mean(X, axis=1, keepdims=True)
    X_c = X - mean

    covariance = (1/(299))*((X_c)@(X_c.T))
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    sorted_idx = np.argsort(eigenvalues)[::-1] #returns sorted indexes.
    eigenvalues, eigenvectors = eigenvalues[sorted_idx], eigenvectors[:, sorted_idx]

    total_variance=np.sum(eigenvalues)
    sum=0
    p=0
    for i in eigenvalues:
        sum+=i
        p+=1
        if (sum/total_variance>=minimum):
            break

    if (num_com):
        p=2
        #print("P is:", p)
    #print("p",p)

    # Reduce dimensionality
    U_p = eigenvectors[:, :p]
    Y = (U_p.T@X_c)

    #print("U_p: ", U_p.shape)

    return Y, U_p, mean


### Functions for FDA

def compute_scatter_matrices(X, y):
    overall_mean = np.mean(X, axis=0)
    overall_mean = overall_mean.reshape(-1, 1)

    #X_n = X.T-overall_mean
    # S_t=  (X_n) @ (X_n.T)

    mean, covariance = mean_and_cov(X, y, 1)

    S_b = np.zeros((784, 784))
    S_w = np.zeros((784,784))

    for i in [0,1,2]:
        mean[i]= mean[i].reshape(-1,1)
        z = mean[i] - overall_mean
        S_b += (100* (z @ z.T))

    #print(X.shape)

    #print("S_b: ", S_b.shape)

    for i in range(0,300):
        z = X[i].reshape(-1,1)
        #print ("z:", z.shape, mean[y[i]].shape)
        z = z-mean[y[i]]
        S_w += (z @ z.T)

    #print("S_w:", S_w.shape)

    return S_w, S_b
    #return S_w, S_t-S_w

def fda(X, y, num_components=2):
    S_W, S_B = compute_scatter_matrices(X, y)

    #S_W = S_W + (0.001 * np.eye(S_W.shape[0]))
    #print(S_W.shape)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W) @ (S_B))
    
    # Sort eigenvectors by eigenvalues in descending order
    sorted_indices = np.argsort(abs(eigenvalues))[::-1]
    W = eigenvectors[:, sorted_indices[:num_components]]

    #print("W",W.shape, X.shape)

    Y_fda = W.T @ X.T
    return Y_fda, W



### Functions used for accuracy of lda and pda

def calculate_discriminant(mean, covariance, x):
    """
    as prior probability is same ignoring that and giving qda
    """
    #calculating the determinant and inverse
    determinant = np.linalg.det(covariance)
    inverse = np.linalg.inv(covariance)

    #print("X", x.shape, mean.shape)

    X=x-(mean.reshape(-1,1))
    return -np.log(determinant) - (X.T @ (inverse @ X))


def mean_cov(covariance):
    new_cov = (1/3)*(covariance[0]+covariance[1]+covariance[2])
    for i in [0,1,2]:
        covariance[i]=new_cov
    
    return covariance

def accuracy(x, y, mean, covariance, lda=False):
    if (lda):
        covariance = mean_cov(covariance)
    correct=0
    for i in range (0,300):
        discriminant=[]
        for j in [0,1,2]:
            #print("mean[j]", mean[j].shape)
            discriminant.append(calculate_discriminant(mean[j], covariance[j], x.T[i].reshape(-1,1)))
        discriminant=[np.real(a) for a in discriminant]
        if (np.argmax(discriminant)==y[i]):
            correct+=1

    if (lda):
        print("LDA accuracy: ", correct/3, "%", sep="")
    else:
        print("QDA accuracy: ",correct/3, "%", sep="")
    return correct/3


### Function used for plotting

def plot(X, y, title):
    plt.figure(figsize=(8, 6))
    for label in [0,1,2]:
        plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Class {label}", alpha=0.7)
    
    plt.legend()
    plt.title(title)
    plt.show()



### start of main code
# Step 1: Preprocess Data
x_train, y_train, x_test, y_test = preprocess_data()

# Step 2: Compute MLE Estimates
mle_mean, mle_covariance = mean_and_cov(x_train, y_train)

# Step 3: Apply PCA (retain 95% variance)
Y_pca, U_p, mean_pca = pca(x_train.T)
x_c_test = x_test.T - mean_pca

z_test = (U_p.T)@((x_c_test))
#z_test = (U_p.T) @ (x_test.T)

### means and covariacnce which would be used for the discriminant analysis after PCA
means_pca, covariance_pca = mean_and_cov(Y_pca.T, y_train, 1)

# Step 5(ii): Accuracy of LDA and QDA after PCA
print("PCA with 95% variance:")
print("Test Data:")
accuracy(z_test, y_test, means_pca, covariance_pca)
accuracy(z_test, y_test, means_pca, covariance_pca, True)
print("Train Data")
accuracy(Y_pca, y_train, means_pca, covariance_pca)
accuracy(Y_pca, y_train, means_pca, covariance_pca, True)
print()

plot(Y_pca.T, y_train, "PCA Projection with 95% variance maintained")

# Step 4: Apply FDA
X_train_fda, W_fda = fda(x_train, y_train)
X_test_fda = W_fda.T @ x_test.T

### means and covariance used for determinant analysis after FDA
mean_fda, covariance_fda = mean_and_cov(X_train_fda.T, y_train, 1)
print("After FDA")
# Step 5(i): Accuracy of LDA and QDA on test and train set
print("Test Data:")
accuracy(X_test_fda, y_test, mean_fda, covariance_fda)
accuracy(X_test_fda, y_test, mean_fda, covariance_fda, True)

print ("Train Data:")
accuracy(X_train_fda, y_train, mean_fda, covariance_fda)
accuracy(X_train_fda, y_train, mean_fda, covariance_fda, True)
print()

plot(X_train_fda.T, y_train, "FDA Projection")

# Step 5(iii): Apply PCA (retain 90% variance)
Y_pca, U_p, mean_pca = pca(x_train.T, 0.90)
x_c_test = x_test.T - mean_pca
z_test = (U_p.T)@((x_c_test))
#z_test = (U_p.T) @ (x_test.T)

means_pca, covariance_pca = mean_and_cov(Y_pca.T, y_train, 1)

print("PCA with 90% variance:")
print("Test Data:")
accuracy(z_test, y_test, means_pca, covariance_pca)
accuracy(z_test, y_test, means_pca, covariance_pca, True)
print("Train Data")
accuracy(Y_pca, y_train, means_pca, covariance_pca)
accuracy(Y_pca, y_train, means_pca, covariance_pca, True)
print()

plot(Y_pca.T, y_train, "PCA Projection with 90% variance maintained")



# Step 5(iv): Apply PCA (using only 2 components)
Y_pca, U_p, mean_pca = pca(x_train.T, 0.90, True)
x_c_test = x_test.T - mean_pca
z_test = (U_p.T)@((x_c_test))
#z_test = (U_p.T) @ (x_test.T)

means_pca, covariance_pca = mean_and_cov(Y_pca.T, y_train, 1)

print("First 2 principal components:")
print("Test Data:")
accuracy(z_test, y_test, means_pca, covariance_pca)
accuracy(z_test, y_test, means_pca, covariance_pca, True)
print("Train Data")
accuracy(Y_pca, y_train, means_pca, covariance_pca)
accuracy(Y_pca, y_train, means_pca, covariance_pca, True)
print()

plot(Y_pca.T, y_train, "PCA Projection with p=2")