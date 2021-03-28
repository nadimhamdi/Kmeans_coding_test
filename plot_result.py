from plotnine import *
from kmeans import KMeans
import pandas as pd
import argparse
from random import random

#N_CLUSTER = 6
#N_POINTS = 100
#N_DIMENS = 2


def rand_point(n_dimens):
    return tuple(random() for _ in range(n_dimens))

if __name__ == "__main__":



    my_parser = argparse.ArgumentParser(description='give the number of cluster, points and dimensions')

    # Add the arguments

    my_parser.add_argument('--cluster','-c', action='store', type=int, required=True,help='number of cluster', default=6)
    my_parser.add_argument('--points','-p', action='store', type=int, required=True,help='number of points', default=1000)
    my_parser.add_argument('--dimens','-d', action='store', type=int, required=True,help='dimensions', default=2)

    # Execute the parse_args() method
    args = my_parser.parse_args()

    N_CLUSTER = args.cluster
    N_POINTS = args.points
    N_DIMENS = args.dimens


    points = [rand_point(N_DIMENS) for _ in range(N_POINTS)]
    print(points)
    kmeans = KMeans(N_CLUSTER).fit(points)
    predictions = kmeans.predict(points)
    table = pd.DataFrame(points)
    table["tag"] = predictions
    table.rename(columns={0: "x", 1: "y"}, inplace=True)
    centroids = kmeans.centroids
    print(centroids)
    table2 = pd.DataFrame(centroids)
    _=[]
    for i in range(N_CLUSTER):
        _.append(i)
    table2["tag"] =_ 
    table2.rename(columns={0: "x", 1: "y"}, inplace=True)
    
    plot = ggplot(aes(x="x", y="y"), table) + geom_point(aes(color="factor(tag)"))+ggtitle("Clustred Data")
    plot.save(filename="Clustred_Data.png", height=5, width=5, units="in", dpi=300)

    plot = ggplot(aes(x="x", y="y"),table2) + geom_point(aes(color="factor(tag)"))+ggtitle("Centroid's coordinates")
    plot.save(filename="Centroid's_coordinates.png", height=5, width=5, units="in", dpi=300)


