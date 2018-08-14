# Dimensionality Reduction and Its Application in Data Mining

This repository is a Master course project for CS5234 at NUS.

In this project we present a brief introduction to PCA, random projection(RP) and sparse random projection(SRP), 
and empirically studied their performance on k-means and kNN on text and image data. 
All three dimensionality reduction methods provide decent performance, which is close to applying k-means and kNN on original data. 
RP outperforms SRP in most cases, and is also able to outperform PCA in certain cases. 
Both RP and SRP are significantly faster than PCA when the dataset and dimensionality are large, 
and SRP runs fastest in most cases due to its sparsity.
Our experimental evaluation shows that these dimensionality reduction techniques are beneficial in applications 
where the pairwise distance between data points are important. 
By giving decent performance and being much more efficient than PCA, 
RP and SRP are good alternatives to the statistically optimal method PCA, 
especially when the dimensionality of the dataset is large.

All experiments are done in `random_projection.ipynb`.
For more information the basics of PCA and random projection, and on the experimental results of this project, please refer to **Report.md**.
