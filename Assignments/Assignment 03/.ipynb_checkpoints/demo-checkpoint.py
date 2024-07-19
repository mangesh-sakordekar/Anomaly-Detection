import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from optigrid import Optigrid

if __name__ == "__main__":
    # First, generate two separate normal distributions and noise
    normal1_mean = [-5, -5, -5]
    normal1_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal1_samples = 10000
    normal1 = np.random.multivariate_normal(mean=normal1_mean, cov=normal1_cov, size=normal1_samples)

    normal2_mean = [0, 0, 0]
    normal2_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal2_samples = 20000
    normal2 = np.random.multivariate_normal(mean=normal2_mean, cov=normal2_cov, size=normal2_samples)
    
    normal3_mean = [-5, 5, 5]
    normal3_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal3_samples = 10000
    normal3 = np.random.multivariate_normal(mean=normal3_mean, cov=normal3_cov, size=normal3_samples)
    
    normal4_mean = [5, 5, 5]
    normal4_cov = [[1, 0, 0], [0, 1, 0], [0, 0, 0.05]]
    normal4_samples = 10000
    normal4 = np.random.multivariate_normal(mean=normal4_mean, cov=normal4_cov, size=normal4_samples)

    noise_low = [-10, -10, -10]
    noise_high = [10, 10, 10]
    noise_samples = 10000
    noise = np.random.uniform(low=noise_low, high=noise_high, size=(noise_samples, 3))

    data = np.concatenate((normal1, normal2, normal3, normal4))#, noise))
    
    # Weight the samples from the first population twice as high
    weights = np.array([5] * normal1_samples + [4] * normal2_samples + [3] * normal3_samples + [2] * normal4_samples)

    # Now we want to standard scale our data. Although it is not necessary, it is recommended for better selection of the parameters and uniform importance of the dimensions.
    data_scaled = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    # Next, chose the parameters
    d = 3 # Number of dimensions
    q = 3 # Number of cutting planes per step
    noise_level = 0.1
    max_cut_score = 0.3
    bandwidth = 0.1

    # Fit Optigrid to the data
    optigrid = Optigrid(d=d, q=q, max_cut_score=max_cut_score, noise_level=noise_level, kde_bandwidth=bandwidth, verbose=True)
    optigrid.fit(data_scaled, weights=weights)
    ### Output: 
    ###     In current cluster: 47.08% of datapoints
    ###     In current cluster: 52.92% of datapoints
    ###     Optigrid found 2 clusters.

    for i, cluster in enumerate(optigrid.clusters):
        cluster_data = np.take(data, cluster, axis=0) # Clusters are stored as indices pointing to the original data
        print("Cluster {}: Mean={}, Std={}".format(i, np.mean(cluster_data, axis=0), np.std(cluster_data, axis=0)))
    ### Output: 
    ###     Cluster 0: Mean=[-5.03474967 -3.3355985   0.6569438 ], Std=[1.79700025 4.11403245 3.33377444]
    ###     Cluster 1: Mean=[ 4.92505754  0.05634452 -0.62898176], Std=[1.92237979 3.49116619 3.46671477]

    
    # Draw a 10 values from both normals and score it with optigrid after normalization
    sample_size = 10
    sample1 = np.random.multivariate_normal(normal1_mean, normal1_cov, sample_size)
    sample2 = np.random.multivariate_normal(normal2_mean, normal2_cov, sample_size)
    sample3 = np.random.multivariate_normal(normal3_mean, normal3_cov, sample_size)
    sample4 = np.random.multivariate_normal(normal4_mean, normal4_cov, sample_size)
    sample = np.concatenate((sample1, sample2,sample3, sample4))#, sample3))
    sample = (sample - np.mean(data)) / np.std(data)

    result = optigrid.score_samples(sample)
    print(result)
    
    
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data_scaled[:,0], data_scaled[:,1], data_scaled[:,2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    cutt_dim = np.zeros(shape = (100,100))
    for cutting_plane in optigrid.cutting_planes:
        print(cutting_plane)
        model_surface = np.zeros(shape=(100, 3))
        dimension = cutting_plane[1]
        cutt_dim[:,:] = cutting_plane[0]
        if (dimension == 0):
            xx = np.linspace(np.min(data_scaled[:,1]), np.max(data_scaled[:,1]), 100)
            xy = np.linspace(np.min(data_scaled[:,2]), np.max(data_scaled[:,2]), 100)
            xx,xy = np.meshgrid(xx, xy)
            ax.plot_surface(cutt_dim, xx, xy, color = 'purple')
        elif(dimension == 1):
            xx = np.linspace(np.min(data_scaled[:,0]), np.max(data_scaled[:,0]), 100)
            xy = np.linspace(np.min(data_scaled[:,2]), np.max(data_scaled[:,2]), 100)
            xx,xy = np.meshgrid(xx, xy)
            ax.plot_surface(xx, cutt_dim, xy, color = 'green')
        else:
            xx = np.linspace(np.min(data_scaled[:,0]), np.max(data_scaled[:,0]), 100)
            xy = np.linspace(np.min(data_scaled[:,1]), np.max(data_scaled[:,1]), 100)
            xx,xy = np.meshgrid(xx, xy)
            ax.plot_surface(xx, xy, cutt_dim, color='red')        
        
    plt.show()
        
    ### Output: 
    ###     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ### The first ten values belong to the zeroth cluster and the latter ten to the second cluster as expected