# Dimensionality Reduction and Its Application in Data Mining

**Team members:**

* Yang Ruizhi, A0105733M
* Chen Shaozhuang, A0134531R
* Li Yuanda, A0078501J

## Abstract

Dimensionality reduction is a commonly used technique for processing high dimensional data. By projecting high dimensional data onto a lower-dimensional space, many machine learning tasks can run significantly faster. However, many classic dimensionality reduction techniques such as principle component analysis (PCA) are expensive to perform. In some cases, the benefit in time complexity that a machine learning algorithm gained from lower dimensional data can be less than the cost of performing PCA. As another powerful technique for dimensionality reduction, random projection is much more efficient in computational time, and has the nice property of almost preserving Euclidean distance when certain condition is satisfied. In this report, we give an introduction to both PCA and random projection, and present experimental results on how they can boost the performance of two classic machine learning algorithms, which are k-means clustering and k-nearest-neighbor classification.

## 1. Introduction

Many data mining applications deal with high-dimensional data. For example, when dealing with large amount of text and image data, it is common to represent the data in a high-dimensional vector space. Some classic algorithms for data mining and predictive analysis including k-means clustering and k-nearest-neighbor (kNN) classification become computationally expensive when applied to such high-dimensional data. Therefore, it is desirable to perform dimensionality reduction with low distortion on the original data, before further mining and analysis are applied.

Principle component analysis (PCA) is one of the most widely used technique to perform dimensionality reduction. It is optimal, in the sense that it gives minimized mean square error over all projections onto a space with the same dimension. However, PCA is computationally expensive, and hence is not a good match for high-dimensional data sets. Fortunately, another dimensionality reduction technique, random projection has been found to be much more efficient than PCA. According to Johnson-Lindenstrauss lemma [1], the reduced data set can be sufficiently accurate (in the sense of preserving pairwise Euclidean distance) by applying random projection on the high dimensional data. The sparse version of random projection can be even cheaper by introducing sparsity into the random matrix used in projection.

In this report, we give a introduction on three dimensionality reduction methods, which are PCA, Gaussian random projection, and sparse random projection. We also briefly describe some of their properties, and then introduce Johnson-Lindenstrauss lemma, which gives performance guarantee for random projection. 

Then we empirically study the performance and time complexity of the three dimensionality reduction methods by applying them on high-dimensional text and image datasets, and verify that the distances between data points can be nearly preserved. In addition, as J-L lemma only gives a worst case bound for the choice of reduced dimension, it is also interesting to experimentally study how the dimension after reduction affect the performance of random projection on the datasets.

After empirically studying the two dimensionality reduction techniques, we apply some classic data mining algorithms on the original high-dimensional data sets and low-dimensional data sets, and compare the results and time complexity of these algorithms. Since random projection has the nice property of nearly preserving interpoint distance of the data, we will be particularly interested in experimenting with algorithms which utilize interpoint distances. In particular, k-means clustering and kNN will be studied in this report.

We will use two real-world datasets on text and image respectively. The text dataset we use is the *20 newsgroups dataset*, which comprises around 18000 newsgroups posts on 20 topics. The image dataset is the *Olivetti faces dataset* from AT&T Laboratories Cambridge, which consists of 400 64*64 images of 40 distinct subjects. Both of these two datasets are high dimensional, and Euclidean distance is important to cluster the data into different set of topics or subjects.

The rest of this report is organized as follows: Section 2 gives an introduction on PCA and random projection. In Section 3, we empirically study the performance of PCA, random projection, and sparse random projection, by applying these dimensionality reduction techniques to text and image datasets. Then in Section 4, we empirically study the performance of k-means and kNN on the data sets before and after dimensionality reduction, and hence understand how dimensionality reduction can help with practical data mining tasks. Finally Section 5 concludes this report.

## 2. Methods for Dimensionality Reduction

In this section, we give an brief introduction to PCA and random projection. We consider that the original data is $d$-dimensional, and we want to reduce the dimensionality to $k$, where $k <<d$. We use $X_{d \times N}$ to denote the original dataset of $N$ $d$-dimensional observations.

#### 2.1 Principle Component Analysis

To perform PCA, we first consider the eigenvalue decomposition of the data covariance matrix $\frac{1}{N-1}XX^T$:
$$
\frac{1}{N-1} XX^T = E \Lambda E^T
$$
where columns of $E$ is eigenvectors corresponding to the respective eigenvalues in $\Lambda$. If a reduced dimension of $k$ is desired, then the $N$ $k$-dimensional data matrix after dimensionality reduction using PCA can be obtained by computing:
$$
X_{PCA} = E_k^T X
$$
where the $E_k$ of dimension $d \times k$ contains the $k$ eigenvectors corresponding to the $k$ largest eigenvalues in $\Lambda$. By performing PCA for dimensionality reduction, the mean squared error introduced in the projection is minimized over all possible projections onto a $k$-dimensional space.

Despite its optimality in a sense, one disadvantage of PCA is it is expensive to compute. The time complexity of performing PCA is $O(d^2N+d^3)$.

#### 2.2 Random Projection and Its Sparse Variant

Gaussian random projection (we will call it random projection in this report) is done using:
$$
X_{RP} =RX
$$
Where $X$ of dimension $d\times N$ is the original data matrix, $R$ of dimension $k \times d$ is the projection matrix, and $X_{RP}$ has dimension $k\times N$.

The key of random projection is how to get the random matrix $R$. In a Gaussian random projection, each entry $r_{ij}$ of $R$ follows $i.i.d$ Gaussian distribution $\mathcal{N}(0, \frac{1}{k})$. The time complexity for generating a random matrix and performing the matrix multiplication in random projection is $O(dkN)$.

To further save computational savings in practice, there are many sparse variants of random projection. One common choice is as the following [2]: 
$$
r_{ij}=\sqrt {3} \cdot 
    \begin{cases}
      +1, & \text{with probability}\ 1/6 \\
      0, & \text{with probability 2/3} \\
      -1, & \text{with probability 1/6}
    \end{cases}
$$
where only integers -1, 0, 1 are required in the random projection matrix.

In section 3 and 4, we will empirically study both the normal random projection where $r_{ij}$ sampled from Gaussian distribution, and the sparse random projection shown above.

Despite the simplicity of random projection, it can be shown that if the dimension $k$ is suitably high, then the distances between the data points can be approximately preserved after projection [1]. We will introduce the corresponding Lemma in the next subsection. 

#### 2.3 Johnson-Lindenstrauss Lemma 

The Johnson-Lindenstrauss Lemma Lemma, as stated in [1], :

> For any $0<ϵ<1$ and any integer $N$, let $k$ be a positive integer such that $k≥4(ϵ^2/2−ϵ^3/3)^{−1}log(N)$. Then for any set $V$ of $N$ points in $\mathbb{R}^{d}$, there is a mapping $f$: $\mathbb{R}^d \rightarrow \mathbb{R}^k$ such that for all $u,v∈V$,
> $$
> (1−ϵ)||u−v||^2≤||f(u)−f(v)||^2≤(1+ϵ)||u−v||^2
> $$
>

The J-L Lemma says that with $k$ large enough, there exists a projection to $\mathbb{R}^k$ such that the distance after projection is an $\epsilon$-approximation of the original distance. In one of the proof of J-L Lemma [3], we can get an explicit construction of a random matrix that produces the desired projection. This establishes the main theoretical foundation behind the efficiency of random projection.

## 3. Experimental Performance of Dimensionality Reduction Methods

#### 3.1 Results on Text Data 

To experimentally study dimensionality reduction on text data, we use the 20 newsgroups dataset provided by sklearn API. For each data point in the text dataset, we convert it to a **term frequency–inverse document frequency** (TF-IDF) vector. TF-IDF vectorization is a common way to represent a document as a vector, by reflecting and adjusting the frequency of a word carefully. In the TF-IDF representation of our dataset has a dimensionality of 130107, we reduce the dimensionality to 2000, using random projection, sparse random projection, and PCA respectively.

In this experiment, we are interested in how well various dimensionality reduction methods can reserve pairwise distance. For each dimensionality reduction method, we compare the original pairwise distance of the text vectors with the pairwise distance after dimensionality reduction. In particular, for the ease of visualisation, we take 100 vectors from the vectorized text dataset, and compute their paiwise distance before and after dimensionality reduction. For each pair of vectors, we calculate $(Projected$ $Distance$ $/$ $Original$ $Distance)$, and plot the overall results in a histogram plot.

*Histogram for RP on text:*

![alt text](images\text_rp_hist.png)

*Histogram for SRP on text:*

![alt text](images\text_srp_hist.png)

*Histogram for PCA on text:*

![alt text](images\text_pca_hist.png)



As shown in the graphs above, PCA performs very well on this small dataset of size = 100 and reduced dimension = 2000. This justifies the "optimality" of PCA as mentioned in last section.

RP gives a decent performance, with most ratios are concentrated around 1. SRP gives a worse performance than RP, with a less concentrated histogram. Overall, both RP and SRP shows their cabability in preserving pairwise distance on this text dataset.

#### 3.2 Results on Image Data

The Olivetti faces dataset we use contains 400 images with dimension 4096. Similar to the text dataset, we take 100 image vectors from this dataset, reduce the dimension from 4096 to 500 using various dimensionality reduction methods, and then visualize their performance.

*Histogram for RP on image:*

![alt text](images\image_rp_hist.png)

*Histogram for SRP on image:*

![alt text](images\image_srp_hist.png)

*Histogram for PCA on image:*

![alt text](images\image_pca_hist.png)

Similar to the results in last subsection, PCA gives impressive performance, by almost keeping the pairwise distance exactly. This is due to the small size of the dataset, as well the "optimality" property of PCA.

SRP gives almost the same performance as RP, although RP still outperforms SRP slightly by giving a slightly more concentrated histogram.

To summarise this section, we have seen that PCA performs nearly perfect in preserving pairwise distance on both of our datasets. Both RP and SRP give decent performance, by preserving pairwise distance effectively, although RP performs slightly better than SRP.

## 4. Experimental Performance of K-Means and K-Nearest-Neighbors

#### 4.1 Experimental Settings

In our experiment, we performed K-Means and K-NN algorithms on Text data and Image data. For each algorithm and data type, we applied 3 type of dimension reduction, Random Projection, Sparse Random Project and PCA. Then compare the results before and after dimension reduction.

For the text data, we use the 20 newsgroups dataset provided by sklearn API. The original *20 newsgroups dataset* comprises around 18000 news posts on 20 topics. To conduct our experiment within a rational time limit, we choose 5 topics from the original dataset.  For each data point in the text dataset, we convert it to a TF-IDF vector. In the TF-IDF representation of our dataset, each of the 2599 data points has a length of 41853.

For the image data, we use the Olivetti faces dataset from AT&T. The original dataset  consists of 10 pictures each of 40 individuals. To conduct our experiment faster, we choose 10 individuals' face data, each has 10 pictures.

#### 4.2 Results on K-Means 

K-Means clustering is a common unsupervised learning algorithm, which is used to find groups which have not been explicitly labeled in the data. K-Means algorithm requires a hyperparameter $k$, which is the number of clusters to be considered. We set $k=5$ clustering the text data, and $k=10$ for clustering image data. The algorithm works iteratively to assign each data point to one of the groups based on the features that are provided.

We compare the performance of K-Means in terms of time, homogeneity, completeness, and v-measure. Homogeneity, completeness and V-measure are common measurement metrics for clustering, when the ground truth of the clusters are available. A homogeneity score is higher if the clusters contain only data points which are members of a single class. A completeness score is higher if all the data points that are members of a given class are elements of the same cluster. The harmonic mean of homogeneity and completeness gives V-measure. All experiments have been done for 20 times, and we take the mean as the final result for comparison.

##### 4.2.1 Results on Text Data

In this experiment, we conduct K-Means on 3387 news from 4 topics. The original data has dimension of 10000, and is reduced to 2000 by RP, SRP and PCA. The following table shows the performance results for original data and dimensionality-reduced data. 

|              | Original Text + K-Means | RP + K-Means | SRP + K-Means | PCA + K-Means |
| ------------ | ----------------------- | ------------ | ------------- | ------------- |
| Time         | 77.56s                  | 5.6s +16.89s | 2.7s + 16.92s | 73s + 15.92s  |
| Homogeneity  | 0.467                   | 0.470        | 0.465         | 0.461         |
| Completeness | 0.553                   | 0.562        | 0.544         | 0.556         |
| V-measure    | 0.506                   | 0.517        | 0.501         | 0.504         |

The performance of clustering is almost the same for all the 3 dimensionality reduction methods, with RP performs slightly better than the other two. In terms of running, RP and SRP perform very well, SRP performs best due to the advantage of sparsity. Surprisingly, RP slightly outperforms the original dataset. This might be due to the specific properties of the vectorized representation of the text dataset, as well as some experimental errors. Despite the perfect performance of PCA in Section 3, PCA on this dataset of larger size (3387) is not as impressive, by giving a slightly worse performance than RP.

In terms of the running time, SRP performs the best as expected, due to the sparsity of random matrix. RP is also efficient, by giving a slightly slower performance than SRP. PCA takes much longer time, and the overall time used for PCA + K-Means is more than the time of running K-Means on original data. This verifies the larger time complexity of PCA comparing to RP and SRP.

##### 4.2.2 Results on Image Data

For image data, we conduct clustering task on 100 images from 10 categories. We reduce the dimension from 4096 to 500 using various dimensionality reduction methods. The following table shows the performance results for original data and dimensionality-reduced data. 

|              | Original Image + K-Means | RP + K-Means   | SRP + K-Means  | PCA + K-Means  |
| ------------ | ------------------------ | -------------- | -------------- | -------------- |
| Time         | 0.381s                   | 0.07s + 0.181s | 0.05s + 0.174s | 0.02s + 0.114s |
| Homogeneity  | 0.646                    | 0.639          | 0.599          | 0.580          |
| Completeness | 0.675                    | 0.667          | 0.620          | 0.603          |
| V-measure    | 0.660                    | 0.653          | 0.609          | 0.591          |

Similar to the case on text, the performance of RP + K-means outperforms SRP and PCA, though RP has slightly lower performance than the original data in this case. It is worth noted that PCA has the best time efficiency. The reason might be due to the overhead for creating random matrix that RP and SRP involved. , Since the dataset is quite small in this case, the benefit gained from time complexity is much smaller than the overheads in the cases of RP and SRP.

#### 4.3 Results on K-Nearest-Neighbors

##### 4.3.1 Results on Text Data

We use the *hold-out* method to split the 2599 data points into training set and test set with a proportion of 0.6:0.4. Evaluation is done by training a kNN model on the training set, and testing its performance on the test set. Such splitting and evaluation are done for 20 times, and we take the mean of the results as our evaluation result. The main purpose of this evaluation is to compare the time complexity on different datasets, and roughly compare their performance on test sets, so our evaluation setting should be sufficient, though it might not be optimal for estimating generalization ability of a model.

We use precision, recall, and f1-score as our metrics for evaluating the performance of a classifier. Precision measures the ability of the classifier not to label as positive a sample that is negative, and recall measures the ability of the classifier to find all the positive samples. F1-score is a weighted harmonic mean of the precision and recall. Higher values of the metrics mean better performance.

###### kNN on Original Text Data

The following table shows the performance result of kNN on the original dataset, where each data point has 41853 dimensions:

```
                       precision    recall  f1-score

          alt.atheism       0.61      0.95      0.74
        comp.graphics       0.98      0.71      0.82
              sci.med       0.97      0.68      0.80
talk.politics.mideast       0.71      0.99      0.83
   talk.religion.misc       0.93      0.56      0.70

          avg / total       0.84      0.79      0.79
```

The average running time is around 122.48 seconds.

###### kNN on Text Data after RP

The following table shows the performance result of kNN on the dataset after random projection, where each data point has a reduced dimension of 2000:

```
                       precision    recall  f1-score

          alt.atheism       0.66      0.89      0.75
        comp.graphics       0.97      0.70      0.82
              sci.med       0.89      0.79      0.83
talk.politics.mideast       0.77      0.96      0.85
   talk.religion.misc       0.80      0.59      0.68

          avg / total       0.82      0.80      0.80
```

The avarage running time for applying RP is 8.23 seconds. The average running time for performing kNN is 6.15 seconds. The combined time cost is significantly less than running kNN on the original datasets, while the precision, recall, and f1 score are almost the same as the original dataset.

###### kNN on Text Data after SRP

The following table shows the performance result of kNN on the dataset after sparse random projection, where each data point has a reduced dimension of 2000:

```
                       precision    recall  f1-score

          alt.atheism       0.79      0.88      0.83
        comp.graphics       0.72      0.90      0.80
              sci.med       0.93      0.61      0.74
talk.politics.mideast       0.79      0.92      0.85
   talk.religion.misc       0.83      0.66      0.73

          avg / total       0.81      0.80      0.79
```

The avarage running time for applying RP is 3.13 seconds. The average running time for performing kNN is 6.17 seconds. The combined time cost is significantly less than running kNN on the original datasets, and also a bit more efficient than running kNN on the dataset after RP. The precision, recall, and f1 score are a bit lower than RP, although the overall performance is still close to the original dataset.

###### kNN on Text Data after PCA

The following table shows the performance result of kNN on the dataset after performing PCA, where each data point has a reduced dimension of 2000:

```
                       precision    recall  f1-score

          alt.atheism       0.69      0.87      0.77
        comp.graphics       0.98      0.67      0.79
              sci.med       0.99      0.63      0.77
talk.politics.mideast       0.56      0.99      0.72
   talk.religion.misc       0.85      0.50      0.63

          avg / total       0.82      0.74      0.75
```

The average running time for applying PCA is 111.72 seconds. The average running time for performing kNN is 6.20 seconds. The combined time cost is around the same as running kNN on the original datasets, and is much less efficient than running kNN after RP or SRP. The precision, recall, and f1 score are lower than RP, although look still decent.

In summary, the results of running kNN on our text data can be summarized in the following table:

|           | Original Image + K-NN | RP + K-NN     | SRP + K-NN    | PCA + K-NN      |
| --------- | --------------------- | ------------- | ------------- | --------------- |
| Time      | 122.48s               | 8.23s + 6.15s | 3.13s + 6.17s | 111.72s + 6.20s |
| Precision | 0.84                  | 0.82          | 0.81          | 0.82            |
| Recall    | 0.79                  | 0.80          | 0.80          | 0.74            |
| F1-Score  | 0.79                  | 0.80          | 0.79          | 0.75            |

Overall, running kNN on all three datasets after dimensionality reduction gives decent predictions. Possibly due to the property to nearly preserve interpoint distance, RP and SRP performs slightly better than PCA. In terms of time complexity, applying PCA + kNN does not improve much over kNN on original dataset. RP + kNN and SRP + kNN are both much faster than kNN on original dataset. Due to its advantage in sparsity, SRP is more efficient than RP.

##### 4.3.2 Results on Image Data

Similar to the experimental setting above for text data, the 100 images are split into training set and test set with a proportion of 0.6:0.4 using *hold-out* and stratified sampling. Evaluation is done by training a kNN model on the training set, and testing its performance on the test set. Such splitting and evaluation are done for 20 times, and we take the mean of the results as our evaluation result. The following table shows the evaluation results:

|           | Original Image + K-NN | RP + K-NN      | SRP + K-NN     | PCA + K-NN     |
| --------- | --------------------- | -------------- | -------------- | -------------- |
| Time      | 0.381s                | 0.07s + 0.181s | 0.05s + 0.174s | 0.02s + 0.114s |
| Precision | 0.815                 | 0.797          | 0.799          | 0.814          |
| Recall    | 0.715                 | 0.719          | 0.721          | 0.718          |
| F1-Score  | 0.714                 | 0.714          | 0.714          | 0.713          |

All three dimensionality reduction methods give very similar results. As the dataset is small, it is difficult to draw any conclusions. In terms of time complexity, as we have reasoned in Section 4.2.2, the PCA is the fastest in running timei, although it has a larger time complexity.

## 5. Conclusion

In this report we have presented a brief introduction to PCA, RP and SRP, and empirically studied their performance on k-means and k-NN on text and image data. All three dimensionality reduction methods provide decent performance, which is close to applying k-means and k-NN on original data. RP outperforms SRP in most cases, and is also able to outperform PCA in certain cases. Both RP and SRP are significantly faster than PCA when the dataset and dimensionality are large, and SRP runs fastest in most cases due to its sparsity.

Our experimental evaluation shows that these dimensionality reduction techniques are beneficial in applications where the pairwise distance between data points are important. By giving decent performance and being much more efficient than PCA, RP and SRP are good alternatives to the statistically optimal method PCA, especially when the dimensionality of the dataset is large.

## References

[1]	Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz mappings into a Hilbert space. *Contemporary mathematics*, *26*(189-206), 1.

[2]	Achlioptas, D. (2001, May). Database-friendly random projections. In *Proceedings of the twentieth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems* (pp. 274-281). ACM.

[3]	Dasgupta, S., & Gupta, A. (2003). An elementary proof of a theorem of Johnson and Lindenstrauss. *Random Structures & Algorithms*, *22*(1), 60-65.