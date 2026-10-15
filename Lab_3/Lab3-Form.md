Introduction to Data Mining

Lab 3 – Simple Classifiers

## Simplicity first!

In the third class, we are going to learn how to examine some data mining algorithms on datasets using Weka. (See the lecture of class 3 by Ian H. Witten, [1][[1]](#footnote-1))

In this section, we learn how **OneR** (one attribute does all the work) works. Open weather.nominal.arff, run OneR, look at the classifier model, how is it?

-

- Remarks:

Use OneR to build decision tree for some datasets. Compared with ZeroR, how does OneR perform?

|  |  |  |
| --- | --- | --- |
| **Dataset** | **OneR -** **accuracy** | **ZeroR - accuracy** |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |
|  |  |  |

## Overfitting

What is “overfitting”? - **overfitting** occurs when a [statistical model](http://en.wikipedia.org/wiki/Statistical_model) describes [**random error**](http://en.wikipedia.org/wiki/Random_error) or **noise** instead of the underlying relationship, b/c of complex model, noise/error in the data, or unsuitable applied criterion, 🡪 poor prediction. To avoid this, use cross-validation, or pruning... [ref: <http://en.wikipedia.org/wiki/Overfitting>]

Follow the instructions in [1], run OneR on the weather.numeric and diabetes dataset…

Write down the results in the following table: (cross-validation used)

|  |  |  |
| --- | --- | --- |
| **Dataset** | **OneR** | **ZeroR** |
| weather.numeric | Classifier model:  Accuracy: | Classifier model:  Accuracy: |
| weather.numeric w/o outlook att. | Classifier model:  Accuracy: | Classifier model:  Accuracy: |
| diabetes | Classifier model:  Accuracy: | Classifier model:  Accuracy: |
| Diabetes w/ minBucketSize 1 | Classifier model:  Accuracy: | Classifier model:  Accuracy: |

MinBucketSize? -

Remark? -

## Using probabilities

Lecture of Naïve Bayes: [1]

* All attributes contribute equally and independently 🡪 no identical attributes

Follow the instructions in [1] to exame **NaiveBayes** on *weather.nominal*

|  |  |
| --- | --- |
| **Classifier model** | **Performance** |
|  | (how many percent of total instances are classified correctly?) |

## Decision Trees

Lecture of decision trees: [1]

How to calculate entropy and information gain?

Entropy measures the impurity of a collection.

$$Entropy\left(S\right)=-\sum\_{i=1}^{c}p\_{i}log\_{2}p\_{i}$$

Information Gain measures the Expected Reduction in Entropy.

*Info. Gain = (**Entropy of distribution before the split) – (Entropy of distribution after the split)*

$$Gain\left(S,A\right)≡Entropy\left(S\right)-\sum\_{v\in Values\left(A\right)}^{}\frac{\left|S\_{v}\right|}{\left|S\right|}Entropy(S\_{v})$$

Values(*A*) is the set of all possible values for attribute *A* and *Sv* is the subset of *S* for which attribute *A* has value.

Build a decision tree for the weather data step by step:

|  |  |
| --- | --- |
| **Compute Entropy and Info. Gain** | **Selected attribute** |
|  |  |
|  |  |
|  |  |
|  | *Final decision tree* |

Use Weka to examine J48 on the weather data.

## Pruning decision trees

Follow the lecture of pruning decision tree in [1] …

Why pruning? - Prevent overfitting to noise in the data.

In Weka, look at the J48 leaner. What are parameters: minNunObj, confidenceFactor?

Follow the instructions in [1] to run J48 on the two dataset, then fill in the following table:

|  |  |  |
| --- | --- | --- |
| **Dataset** | **J48 (default, pruned)** | **J48 (unpruned)** |
| diabetes.arff |  |  |
| breast‐cancer.arff |  |  |

## Nearest neighbor

Follow the lecture in [1]

“Instance‐based” learning = “nearest‐neighbor” learning

What is k‐nearest‐neighbors (K-NN)? –

Follow the instructions in [1] to run lazy>IBk on the *glass* dataset with k = 1, 5, 20, and then fill its accuracy in the following table:

|  |  |  |  |
| --- | --- | --- | --- |
| **Dataset** | **IBk, k =1** | **IBk, k =5** | **IBk, k =20** |
| Glass |  |  |  |

1. http://www.cs.waikato.ac.nz/ml/weka/mooc/dataminingwithweka/ [↑](#footnote-ref-1)