# TASK
Given a set of two dimensional points `P (e.g. [(1.1, 2.5), (3.4,1.9)...]`; the size of set can be 100s), write a function that calculates simple `K`-means. The expected returned value from the function is 

1. a set of cluster id that each point belongs to, and 
2. coordinates of centroids at the end of iteration.

Given a set of two dimensional points P (e.g. [(1.1, 2.5), (3.4,1.9)...]; the size of set can be100s), write a function that calculates simple K-means. The expected returned value from the function is 1) a set of cluster id that each point belongs to, and 2) coordinates of centroids at the end of iteration.

Although you can write this in any language, we would recommend for you to use python.Please feel free to research and look up any information you need, but please note plagiarism will not be tolerated.You may spend as much time as needed, but as a frame of reference, an hour would be the maximum time frame. If more time is required, please send over the intermediate code at the one hour mark.


## Solution

1. The solution implemented works for 2-dimensional clusters we can change the dimensions .
2. It implements a class Kmeans, that only has the `fit`, `fit_predict` and `predict` methods that the sklearn class considers
3. The file `test_kmeans.py` tests that
4. It tooks **One HOUR** to finish the code

## Scripts

It requires python 3.6 or above and pandas and plotnine.

`N_CLUSTER`(-c) `N_POINTS`(-p) `N_DIMENS`(-d) .



### KMeans

The main test to see if everything is ok.

```bash
    python test_kmeans.py -c 6 -p 1000 -d 2
```
### Plotting

The requirements are given in the requirements.txt

```bash
    pip3 install -r requirements.txt
    python plot_result.py -c 6 -p 1000 -d 2
```

This will output a png image called `Clustred_Data.png` `Centroid's_coordinates.png`



## Thanks

Thank you for the opportunity,

Nadim HAMDI
