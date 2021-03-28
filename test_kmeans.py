from kmeans import KMeans


from random import random

N_CLUSTER = 6
N_POINTS = 10
N_DIMENS = 2


def rand_point(n_dimens):
    return tuple(random() for _ in range(n_dimens))


def main():
    points = list(rand_point(N_DIMENS) for _ in range(N_POINTS))
    kmeans = KMeans(N_CLUSTER).fit(points)
    centroids = kmeans.centroids
    
    print("\n***************Centroid's cordinations******************\n",centroids)

    predictions = kmeans.predict(points) 
    
    print("\n***************Predicted_points******************\n",predictions)

    assert all(isinstance(pred, int) and 0 <= pred < N_CLUSTER for pred in predictions)

if __name__ == "__main__":
    main()
