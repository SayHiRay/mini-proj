# Dimensionality Reduction and Its Application in Data Mining

**Team members:**

* Yang Ruizhi, A0105733M
* Chen Shaozhuang, A0134531R
* Li Yuanda, A0078501J

This interim report is a part of our final report for this MiniProject. While all the survey of related theories and most of our experiments have already been done, we do not have the time yet to put in all of them. Instead, we give the full introduction of our project in Section 1, and some experimental results on applying **kNN** in Section 4. We hope this interim report could give a good sense on the topics and approaches involved in our project, and we are thrilled to present a complete and self-contained final report in next week.

## Abstract

Dimensionality reduction is a commonly used technique for processing high dimensional data. By projecting high dimensional data onto a lower-dimensional space, many machine learning tasks can run significantly faster. However, many classic dimensionality reduction techniques such as principle component analysis (PCA) are expensive to perform. In some cases, the benefit in time complexity that a machine learning algorithm gained from lower dimensional data can be less than the cost of performing PCA. As another powerful technique for dimensionality reduction, random projection is much more efficient in computational time, and has the nice property of almost preserving Euclidean distance when certain condition is satisfied. In this report, we give an introduction to both PCA and random projection, and present experimental results on how they can boost the performance of two classic machine learning algorithms, which are k-means clustering and k-nearest-neighbor classification.

## 1. Introduction

Many data mining applications deal with high-dimensional data. For example, when dealing with large amount of text and image data, it is common to represent the data in a high-dimensional vector space. Some classic algorithms for data mining and predictive analysis including k-means clustering and k-nearest-neighbor (kNN) classification become computationally expensive when applied to such high-dimensional data. Therefore, it is desirable to perform dimensionality reduction with low distortion on the original data, before further mining and analysis are applied.

Principle component analysis (PCA) is one of the most widely used technique to perform dimensionality reduction. It is optimal, in the sense that it gives minimized mean square error over all projections onto a space with the same dimension. However, PCA is computationally expensive, and hence is not a good match for high-dimensional data sets. Fortunately, another dimensionality reduction technique, random projection has been found to be much more efficient than PCA. According to Johnson-Lindenstrauss lemma, the reduced data set can be sufficiently accurate (in the sense of preserving pairwise Euclidean distance) by applying random projection on the high dimensional data. The sparse version of random projection can be even cheaper by introducing sparsity into the random matrix used in projection.

In this report, we give a introduction on two dimensionality reduction methods - PCA and random projection, and briefly describe some of their properties. Then we introduce Johnson-Lindenstrauss lemma, which gives performance guarantee for random projection. A sparse version of random projection will also be introduced.

Then we empirically study the performance and time complexity of PCA and random projection by applying them on high-dimensional text and image datasets, and verify that the distances between data points can be nearly preserved. In addition, as J-L lemma gives a worst case bound for the choice of reduced dimension, it is also interesting to experimentally study how the dimension after reduction affect the performance of random projection on the datasets.

After empirically studying the two dimension reduction techniques, we apply some classic data mining algorithms on the original high-dimensional data sets and low-dimensional data sets, and compare the results and time complexity of these algorithms. Since random projection has the nice property of nearly preserving interpoint distance of the data, we will be particularly interested in experimenting with algorithms which utilize interpoint distances. In particular, k-means clustering and kNN will be studied in this report.

We will use two real-world datasets on text and image respectively. The text dataset we use is the *20 newsgroups dataset*, which comprises around 18000 newsgroups posts on 20 topics. The image dataset is the *Olivetti faces dataset* from AT&T Laboratories Cambridge, which consists of 400 64*64 images of 40 distinct subjects. Both of these two datasets are high dimensional, and Euclidean distance is important to cluster the data into different set of topics or subjects.

The rest of this report is organized as follows: Section 2 gives an introduction on PCA and random projection. In Section 3, we empirically study the performance of PCA, random projection, and sparse random projection, by applying these dimensionality reduction techniques to text and image datasets. Then in Section 4, we empirically study the performance of k-means and kNN on the data sets before and after dimensionality reduction, and hence understand how dimensionality reduction can help with practical data mining tasks. Section 5 discusses some of the pros and cons of PCA and random projection. Finally Section 6 concludes this report.

## 2. Methods for Dimensionality Reduction

#### 2.1 Principle Component Analysis

#### 2.2 Random Projection and Its Sparse Variant

#### 2.3 Johnson-Lindenstrauss Lemma 

## 3. Experimental Performance of Dimensionality Reduction Methods

#### 3.1 Results on Text Data 

```
Projected 500 samples from 130107 to 1000 in 0.523s
Random matrix with size: 4.329MB
Mean distances rate: 1.00 (0.10)
Projected 500 samples from 130107 to 2000 in 1.009s
Random matrix with size: 8.651MB
Mean distances rate: 1.04 (0.08)
```

![alt text](images\RP_TXT_1.png)

![alt text](images\RP_TXT_2.png)

![alt text](images\RP_TXT_3.png)

![alt text](images\RP_TXT_4.png)

#### 3.2 Results on Image Data

```
Projected 400 samples from 4096 to 1000 in 0.135s
Random matrix with size: 0.769MB
Mean distances rate: 0.99 (0.04)
Projected 400 samples from 4096 to 2000 in 0.261s
Random matrix with size: 1.530MB
Mean distances rate: 0.99 (0.03)
```



![alt text](images\RP_IMG_1.png)

![alt text](images\RP_IMG_2.png)

![alt text](images\RP_IMG_3.png)

![alt text](images\RP_IMG_4.png)

## 4. Experimental Performance of K-Means and K-Nearest-Neighbors

#### 4.1 Results on K-Means 

##### Results on Text Data

###### K-Means on Original Image Data

```
Homogeneity: 0.467
Completeness: 0.553
V-measure: 0.506
```

###### K-Means on Image Data after RP

```
Homogeneity: 0.479
Completeness: 0.562
V-measure: 0.517
```

###### K-Means on Image Data after SRP

```
Homogeneity: 0.465
Completeness: 0.544
V-measure: 0.501
```

###### K-Means on Image Data after PCA

```
Homogeneity: 0.461
Completeness: 0.556
V-measure: 0.504
```

##### Results on Image Data

##### 

###### kNN on Original Image Data

```
Homogeneity: 0.646
Completeness: 0.675
V-measure: 0.660
```

######  kNN on Image Data after RP

```
Homogeneity: 0.639
Completeness: 0.667
V-measure: 0.653
```

######  kNN on Image Data after SRP

```
Homogeneity: 0.599
Completeness: 0.620
V-measure: 0.609
```

######  kNN on Image Data after PCA

```
Homogeneity: 0.580
Completeness: 0.603
V-measure: 0.591
```

#### 4.2 Results on K-Nearest-Neighbors

##### Results on Text Data

The original *20 newsgroups dataset* comprises around 18000 news posts on 20 topics. To conduct our experiment within a rational time limit, we choose 5 topics from the original dataset, which contain 2599 news. We use the *hold-out* method to split the 2599 data points into training set and test set with a proportion of 0.6:0.4. Evaluation is done by training a kNN model on the training set, and testing its performance on the test set. Such splitting and evaluation are done for 20 times, and we take the mean of the results as our evaluation result. The main purpose of this evaluation is to compare the time complexity on different datasets, and roughly compare their performance on test sets, so our evaluation setting should be sufficient, though it might not be optimal for estimating generalization ability of a model.

For each data point in the dataset, we convert it to a **term frequency–inverse document frequency** (TF-IDF) vector. TF-IDF vectorization is a common way to represent a document as a vector, by reflecting and adjusting the frequency of a word carefully. In the TF-IDF representation of our dataset, each of the 2599 data points has a length of 41853.

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

Overall, running kNN on all three datasets after dimensionality reduction gives decent predictions. Possibly due to the property to nearly preserve interpoint distance, RP and SRP performs slightly better than PCA. In terms of time complexity, applying PCA + kNN does not improve much over kNN on original dataset. RP + kNN and SRP + kNN are both much faster than kNN on original dataset. Due to its advantage in sparsity, SRP is more efficient than RP.

##### Results on Image Data

###### kNN on Original Text Data

```
Avg precision		Avg Recall			Avg f1
(0.8145846819846817, 0.71450000000000002, 0.71391964285714304)
```

###### kNN on Text Data after RP

```
Avg precision		Avg Recall	Avg f1
(0.79705135212010236, 0.71875, 0.71385948336793925)
```

######  kNN on Text Data after SRP

```
Avg precision		Avg Recall			Avg f1
(0.7992482836607836, 0.72074999999999989, 0.71426363224194123)
```

######  kNN on Text Data after PCA

```
Avg precision		Avg Recall			Avg f1
(0.81424647089022106, 0.71825000000000006, 0.71345610879058996)
```

## 5. Pros and Cons of Dimensionality Reduction Methods

## 6. Conclusion

## References