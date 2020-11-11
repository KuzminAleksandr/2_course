#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import combinations

__author__ = 'Aleksandr Kuzmin -- Lomonosov Moscow State University'
__all__ = ['KMeans']

class Kmeans:
    
    def __init__(self, 
                 num_clusters = 3,
                 max_iter = -1,
                 similiarity = "euclidian",
                 auto_clusters = False,
                 init = "uniform") -> None:
        """
        Numpy implementation of сlustering algorithm - K-means (Vanila).
        """
        """
        Parameters
        
        ---------------------------
        
        num_clusters(int): number of clusters in data
        By default: 3 clusters
        
        max_iter(int, optional): maximal number of iterations
        By default: infinity iterations before converges
        
        similiarity(str, optional): similiarity function
        By default: "euclidian"
        It is also possible to use one of the next: ["euclidian", "cosine-distance", "manhattan"]
        
        auto_clusters(bool, optional): whether algorithm try to find and set optimal number of clusters, optional
        
        """
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.auto_clusters = auto_clusters
        self.similiarity = None
        self.centroids = None
        self.metric = None
        self.last_fit_dists = None
        self.labels_stack = []
        self.centroids_stack = []
        
        similiarities = ["euclidian", "cosine-distance", "manhattan"]
        sim_funcs = [self.euclidian_distance, self.cosine_distance, self.manhattan_distance]
        assert similiarity in similiarities, "Incorrect similiarity function. Choose one of the next: {}".format(similiarities)
        self.similiarity = dict(zip(similiarities, sim_funcs))[similiarity]
                
    
    def get_params(self):
        return {"num_clusters": self.num_clusters, 
                "max_iter": self.max_iter, 
                "alpha": self.alpha,
                "similiarity": self.similiarity, 
                "lr": self.lr}
    
    
    def init_centroids(self,
                       X: np.ndarray) -> np.ndarray:
               
        return X[np.random.choice(range(len(X)), self.num_clusters, replace=False), :]
    
    def set_auto_clusters(self,
                          X: np.ndarray) -> list:
        
        metric = []
        for i in range(2, 15):
            self.num_clusters = i
            self.auto_clusters = False
            self.fit(X)

            F_1 = 0
            for m, n in combinations(list(range(self.num_clusters)), r=2):
                F_1 += np.linalg.norm(self.centroids[m, :] - self.centroids[n, :], ord=2)
            F_0 = 0
            for k in range(self.num_clusters):
                F_0 += np.mean(np.linalg.norm(X[self.labels==k, :] - self.centroids[k, :], axis=1), axis=0)
            metric.append(F_0  * self.num_clusters * (self.num_clusters - 1) / (2 * F_1))
            
        return metric
    
    def euclidian_distance(self,
                           X: np.ndarray,
                           dists: np.ndarray) -> np.ndarray:
        
        for i in range(self.num_clusters):
                    dists[:, i] = np.linalg.norm(np.abs(X - self.centroids[i, :]), axis=1)
        return np.argmin(dists, axis=1)
        
    
    def manhattan_distance(self,
                           X: np.ndarray,
                           dists: np.ndarray) -> np.ndarray:
        
        for i in range(self.num_clusters):
                    dists[:, i] = np.sum(np.abs(X - self.centroids[i, :]), axis=1)   
        return np.argmin(dists, axis=1)
    
    
    def cosine_distance(self, 
                        X: np.ndarray,
                        dists: np.ndarray) ->np.ndarray:
        eps = 0.01
        dists = np.dot(X, self.centroids.T)
        centroid_norm = np.linalg.norm(self.centroids, axis=1)
        X_norm = np.linalg.norm(X, axis=1).T
        dists /= X_norm.reshape(-1, 1) 
        dists /= centroid_norm.reshape( 1, -1) 
        dists -= 1
        dists *= -1
        return np.argmin((dists), axis=1)
    
        
    def set_centroids(self,
                      X: np.ndarray,
                      labels: np.ndarray) -> None:

        for i in range(self.num_clusters):
            if len(X[labels == i, :]) == 0:
                continue
            self.centroids[i, :] = np.mean(X[labels == i, :], axis=0)

       
                
    def clustering(self, 
                   X: np.ndarray) -> None:
        
        dists = np.zeros((X.shape[0], self.num_clusters))
        it = 0
        while it != self.max_iter:
            it += 1
            pred_centr = self.centroids.copy()
            labels = self.similiarity(X, dists).copy()
            self.labels_stack.append(labels.copy())
            self.set_centroids(X, labels)
            self.centroids_stack.append(self.centroids.copy())    
            
            if it != 1 and np.array_equal(self.labels_stack[it - 1], self.labels_stack[it - 2]):
                self.labels = labels
                break
        
        self.labels = labels
        
                 
    def fit(self,
            X: np.ndarray) -> None:
        
        if self.auto_clusters:
            self.metric = self.set_auto_clusters(X)

        self.centroids = self.init_centroids(X)
        self.clustering(X)
                
    def predict(self,
                X: np.ndarray) -> np.ndarray:
        self.last_fit_dists = np.zeros((X.shape[0], self.num_clusters))
        labels = self.similiarity(X, self.last_fit_dists)
        return labels        


#EOF
