import cv2

import numpy as np 

import math

import matplotlib.pyplot as plt

import random

iterations = 1
def KnnClustering(Dataset, Centroids):
    
    cluster1 = []
    cluster2 = []
    cluster3 = []
    c1x = Centroids.item((0,0))
    c1y = Centroids.item((0,1))
    c2x = Centroids.item((1,0))
    c2y = Centroids.item((1,1))
    c3x = Centroids.item((2,0))
    c3y = Centroids.item((2,1))
    X=[]
    Y=[]
    X1 = []
    Y1 =[]
    X2 = []
    Y2 =[]
    X3 = []
    Y3 =[]
    

    for i in range(Dataset.shape[0]):
        Distance_centroid1 = math.sqrt(((c1x - Dataset.item((i,0))) ** 2) + ((c1y - Dataset.item((i,1))) ** 2))
        Distance_centroid2 = math.sqrt(((c2x - Dataset.item((i,0))) ** 2) + ((c2y - Dataset.item((i,1))) ** 2))
        Distance_centroid3 = math.sqrt(((c3x - Dataset.item((i,0))) ** 2) + ((c3y - Dataset.item((i,1))) ** 2))
        X.append(Dataset.item((i,0)))
        Y.append(Dataset.item((i,1)))
        if Distance_centroid1 < Distance_centroid2 and Distance_centroid1 < Distance_centroid3:
            cluster1.append((Dataset.item((i,0)),Dataset.item((i,1))))
            X1.append(Dataset.item((i,0)))
            Y1.append(Dataset.item((i,1)))
        elif Distance_centroid2 < Distance_centroid3:
            cluster2.append((Dataset.item((i,0)),Dataset.item((i,1))))
            X2.append(Dataset.item((i,0)))
            Y2.append(Dataset.item((i,1)))
        else:
            cluster3.append((Dataset.item((i,0)),Dataset.item((i,1))))
            X3.append(Dataset.item((i,0)))
            Y3.append(Dataset.item((i,1)))

    cluster1 = np.asarray(cluster1)
    cluster2 = np.asarray(cluster2)
    cluster3 = np.asarray(cluster3)
    
    value1 = sum(cluster1)/len(cluster1)
    value2 = sum(cluster2)/len(cluster2)
    value3 = sum(cluster3)/len(cluster3)
    
    new_centroids = np.matrix([value1,value2,value3])
    
    print (new_centroids)

    global iterations

    plotsofknn(X,Y,c1x,c1y,c2x,c2y,c3x,c3y,X1,Y1,X2,Y2,X3,Y3,iterations)
    
    if(new_centroids.sum() == Centroids.sum()):

        return
    else:
        iterations = iterations + 1    
        KnnClustering(Dataset,new_centroids)
    
def plotsofknn(X,Y,c1x,c1y,c2x,c2y,c3x,c3y,X1,Y1,X2,Y2,X3,Y3,iterations):
    plt.scatter(X1,Y1,marker='^',c='r')
    plt.scatter(X2,Y2,marker='^',c='g')
    plt.scatter(X3,Y3,marker='^',c='b')
    plt.scatter(c1x,c1y,c='r')
    plt.scatter(c2x,c2y,c='g')
    plt.scatter(c3x,c3y,c='b')
    string = 'task3_iter_' + str(iterations) + '.jpg'
    plt.savefig(string)
    plt.close()

    return 

def determineCentroids(image,K):
    Centroids = []
    image = np.asarray(image)
    for i in range(K):
        randint_x = random.randint(0,image.shape[0]-1)
        randint_y = random.randint(0,image.shape[1]-1)
        Centroids.append(image[randint_x][randint_y])
    Centroids = np.asarray(Centroids)  
    print(Centroids)  
    return Centroids


def distanceCalculation(A,B):

    distance = math.sqrt((A.item(0) - B.item(0)) ** 2 + (A.item(1) - B.item(1)) ** 2 + (A.item(2) - B.item(2)) ** 2)
    
    return distance

def calculate_new_centroids(key,values):
    
    R,G,B = 0,0,0
    new_centroids = []
    for i in range(values.shape[0]):
        R += values.item((i,0))
        G += values.item((i,1))
        B += values.item((i,2))
    R = R/ len(values)
    G = G/ len(values)
    B = B/ len(values) 
    
    new_centroids.append(round(R))
    new_centroids.append(round(G))
    new_centroids.append(round(B))
    return new_centroids

def imagequantization(image,K,Centroids):
    
    rows,columns,channels = image.shape
    quantized_image = np.zeros(image.shape)
    Centroids_clusters =[]
    hash_table = [[] for _ in range(K)]
    Distance = []
    clusters = dict()
    for i in range(rows):
        for j in range(columns):
            for k in range(len(Centroids)):
                Distance.append(distanceCalculation(Centroids[k], image[i][j]))
            minIndex = Distance.index(min(Distance))
            quantized_image[i][j] = Centroids[minIndex]
            clusters.setdefault(minIndex, [] ).append(np.asarray(image[i][j]))
            Distance = []

    quantized_image = np.asarray(quantized_image)    
    value1 = []
    value2 = []
    for key,item in clusters.items():
            Centroids_clusters.append(calculate_new_centroids(key,np.asarray(item)))

            #value2 += clusters[1]
    #print(len(value2))        
    #sprint(np.asarray(sum(value1)))
    print(Centroids_clusters)
    Centroids_clusters = np.asarray(Centroids_clusters)
    
    if (np.sum(Centroids_clusters) == np.sum(Centroids)):
       
        string = 'task3_baboon' + str(K) + '.jpg'
        cv2.imwrite(string, quantized_image)
        return
    else:
        imagequantization(image,K, Centroids_clusters)    
















if __name__ == "__main__":

    X = np.matrix('5.9, 3.2;4.6, 2.9;6.2, 2.8;4.7, 3.2;5.5, 4.2;5.0, 3.0;4.9, 3.1;6.7, 3.1;5.1, 3.8;6.0, 3.0')
    C = np.matrix('6.2, 3.2; 6.6, 3.7; 6.5, 3.0')
    KnnClustering(X,C)


    ###############Image Quantization ###########################
    image = cv2.imread('baboon.jpg')
    imagequantization(image, 3, determineCentroids(image,3))
    imagequantization(image,5, determineCentroids(image,5))
    imagequantization(image,10, determineCentroids(image,10))
    imagequantization(image,20, determineCentroids(image,20))







