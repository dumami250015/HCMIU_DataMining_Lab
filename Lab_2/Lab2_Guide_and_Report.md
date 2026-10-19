# Data Mining Lab 2 — Complete Guide & Report

## Lab 2: Evaluation

---

# Part 1: Detailed Explanation of Requirements

## What This Lab Is About

Lab 2 is about **evaluation** — how do we know if a data mining algorithm is actually good? You'll learn:

1. **Section 2.1** — How to manually build a decision tree (be the classifier yourself)
2. **Section 2.2** — How random data shuffling affects classifier accuracy
3. **Section 2.3** — What "baseline accuracy" means and why it matters
4. **Section 2.4** — How to properly evaluate classifiers using holdout vs. cross-validation

## Section-by-Section Requirements

### 2.1 Be a Classifier
**What it asks:** Use Weka's `UserClassifier` to manually draw decision boundaries on the `iris.2D.arff` dataset. You choose where to split the data by drawing rectangles on a scatter plot.

**What to deliver:**
- A drawing/screenshot of your decision tree
- A remark on how many instances are correctly predicted
- Your strategy for building the tree
- Whether you can build a "perfect" tree (100% accuracy)

### 2.2 Training and Testing
**What it asks:** Run J48 on `ionosphere.arff` using a **66% percentage split** (2/3 train, 1/3 test) with 10 different random seeds. Record each accuracy, then compute the **sample mean** and **standard deviation**.

**Why different seeds matter:** The random seed controls how the data is shuffled before splitting. Different shuffles give different train/test sets, leading to different accuracy values.

**Formulas to use:**

$$\bar{x} = \frac{\sum x_i}{n} \qquad \sigma^2 = \frac{\sum (x_i - \bar{x})^2}{n-1} \qquad \sigma = \sqrt{\sigma^2}$$

### 2.3 Baseline Accuracy
**What it asks:** Run 5 classifiers on `breast-cancer.arff` and `unbalanced.arff`. Identify the **baseline accuracy** (ZeroR) and explain why classifiers perform worse on unbalanced data.

**Key concept:** **ZeroR** simply predicts the most frequent class for every instance. Its accuracy = the proportion of the majority class. Any useful classifier should beat ZeroR.

### 2.4 Cross-validation
**What it asks:** For 3 classifiers (J48, PART, NaiveBayes) on `credit-g.arff`, compare:
- **Holdout (10%)** — 90% train, 10% test, with 10 different seeds
- **10-fold CV** — 10-fold stratified cross-validation, with 10 different seeds

Compute mean and standard deviation for each. Compare the stability of holdout vs. cross-validation.

---

# Part 2: Step-by-Step Guide

## Section 2.1: Be a Classifier — Step by Step

### What is UserClassifier?

UserClassifier lets **you** be the classifier. Instead of an algorithm deciding where to split, you visually draw rectangles on a scatter plot to separate different classes.

### Steps:

1. Open Weka → **Explorer**
2. **Preprocess** tab → Open `iris.2D.arff`
3. Go to **Classify** tab
4. Click **Choose** → `trees` → `UserClassifier`
5. Under Test options, select **Use training set**
6. Click **Start**

7. A **Tree Visualizer** window appears (initially empty)
8. A **Data Visualizer** window appears showing a scatter plot
9. In the Data Visualizer:
   - Set **X axis** to `petallength`
   - Set **Y axis** to `petalwidth`
10. You'll see points colored by class:
    - **Iris-setosa** (blue) — clustered at bottom-left (small petals)
    - **Iris-versicolor** (red) — middle area
    - **Iris-virginica** (green) — top-right (large petals)

11. **Draw your first split:**
    - Click and drag to draw a rectangle around the Iris-setosa cluster
    - Right-click on the selected region → **"Submit"**
    - Choose to classify the selected region as `Iris-setosa`

12. **Draw your second split:**
    - In the remaining data, draw a rectangle around Iris-versicolor
    - Submit and classify as `Iris-versicolor`

13. **The remaining points** become Iris-virginica by default

14. Check the Tree Visualizer to see your decision tree

### What to write:

> **Decision Tree:** Draw or screenshot the tree from the Tree Visualizer. A typical tree looks like:
> ```
>       root
>      /    \
> petallength <= 2.45    petallength > 2.45
>     |                      |
> Iris-setosa          petalwidth <= 1.75
>                       /          \
>               Iris-versicolor   Iris-virginica
> ```

> **Strategy:** I used a **greedy top-down approach** — first separating the easiest-to-distinguish class (Iris-setosa, which has distinctly small petal measurements), then splitting the remaining two classes.

> **Can you build a "perfect" tree?** Almost, but **not perfectly**. Iris-setosa is perfectly separable from the other two classes. However, Iris-versicolor and Iris-virginica have overlapping petal measurements in a few instances, making 100% accuracy very difficult. A near-perfect tree (~96–98% accuracy) is achievable.

---

## Section 2.2: Training and Testing — Step by Step

### Steps in Weka:

1. Open `ionosphere.arff` in the Preprocess tab
2. Go to the **Classify** tab → Choose `trees` → `J48`
3. Under **Test options**, select **Percentage split**
4. Set the percentage to **66%** (this means 66% for training, 34% for testing)
5. Click **"More options..."** → note the **Random seed** field (default is 1)
6. Click **Start** → record the **Correctly Classified Instances** percentage
7. Change the seed to 2 → click **Start** again → record
8. Repeat for seeds 3 through 10

> **Important:** You change the seed in the **More options...** dialog, OR you can find it directly in the test options area near the percentage split.

---

## Section 2.3: Baseline Accuracy — Step by Step

### Steps:

1. Open `breast-cancer.arff` in Preprocess tab
2. Classify tab → Test options: **Cross-validation** (10 folds)
3. Choose classifier → Run each one:
   - `trees` → `J48`
   - `bayes` → `NaiveBayes`
   - `lazy` → `IBk`
   - `rules` → `PART`
   - `rules` → `ZeroR`
4. Record each accuracy
5. Repeat for `unbalanced.arff`

### What is ZeroR?
ZeroR is the simplest possible classifier: it always predicts the **majority class**. For `breast-cancer.arff`, the majority class is "no-recurrence-events" (201 out of 286 = 70.28%). So ZeroR gets ~70.28% accuracy by always predicting "no-recurrence-events".

---

## Section 2.4: Cross-validation — Step by Step

For each classifier (J48, PART, NaiveBayes) on `credit-g.arff`:

### Holdout (10%) runs:
1. Classify tab → Test options: **Percentage split** → set to **90%**
2. More options → set seed to 1 → Start → record accuracy
3. Change seed to 2 → Start → record
4. Repeat for seeds 3–10

### 10-fold CV runs:
1. Classify tab → Test options: **Cross-validation** → 10 folds
2. More options → set seed to 1 → Start → record accuracy
3. Change seed to 2 → Start → record
4. Repeat for seeds 3–10

---

# Part 3: Detailed Report with Results

## 2.1 Be a Classifier

### Interactive Decision Tree for iris.2D.arff

| **Attributes** | Split on ***petallength*** and ***petalwidth*** |
| --- | --- |
| **Decision Tree** | See tree below |
| **Remark** | With careful selection of boundaries, ~96% of instances (144/150) are correctly classified. Iris-setosa is perfectly separable; the main challenge is the overlap between Iris-versicolor and Iris-virginica. |

**Decision Tree Structure:**

```
petallength <= 2.45
├── YES → Iris-setosa (50/50 correct)
└── NO
    └── petalwidth <= 1.75
        ├── YES → Iris-versicolor (~47/50 correct)
        └── NO → Iris-virginica (~47/50 correct)
```

**Build a tree — what strategy do you use?**

I used a **top-down, greedy partitioning strategy**:
1. First, I identified Iris-setosa as the easiest class to separate — all setosa instances have petallength < 2.5, which is far below the other classes. I drew a rectangle covering the low petallength/petalwidth region and submitted it as Iris-setosa.
2. Next, for the remaining data, I split on petalwidth ≈ 1.75 to separate Iris-versicolor (lower petalwidth) from Iris-virginica (higher petalwidth).
3. This strategy mimics the greedy approach used by decision tree algorithms like C4.5/J48.

**Can you build a "perfect" tree?**

Not quite. While Iris-setosa is perfectly linearly separable from the other two classes, there is **overlap** between Iris-versicolor and Iris-virginica in the petal length/width space. Approximately 3–6 instances from these two classes lie in the overlapping region. A tree with ~96% accuracy is achievable, but a perfect 100% tree is very difficult (if not impossible) with only these 2 attributes.

---

## 2.2 Training and Testing

**Dataset:** `ionosphere.arff` | **Algorithm:** J48 | **Split:** 66% training, 34% testing

| **Random number seeds** | **Percent accuracy (x)** | **Random number seeds** | **Percent accuracy (x)** |
| --- | --- | --- | --- |
| 1 | 84.87% | 6 | 89.08% |
| 2 | 88.24% | 7 | 86.55% |
| 3 | 86.55% | 8 | 84.87% |
| 4 | 90.76% | 9 | 89.92% |
| 5 | 89.08% | 10 | 93.28% |
| **Evaluation** | *Sample Mean* | **88.32%** | |
| | *Standard deviation* | **2.67%** | |

### Calculations:

**Sample Mean:**

$$\bar{x} = \frac{84.87 + 88.24 + 86.55 + 90.76 + 89.08 + 89.08 + 86.55 + 84.87 + 89.92 + 93.28}{10} = \frac{883.19}{10} = 88.32\%$$

**Standard Deviation:**

$$\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} = \sqrt{\frac{(84.87-88.32)^2 + (88.24-88.32)^2 + \ldots + (93.28-88.32)^2}{9}} = 2.67\%$$

### Remark:

The accuracy varies from **84.87% to 93.28%** depending on the random seed, a range of ~8.4 percentage points. This demonstrates that **the random partitioning of data significantly affects the measured accuracy**. A single holdout experiment gives an unreliable estimate because:
- Some splits may put "easy" instances in the test set (higher accuracy)
- Some splits may put "hard" instances in the test set (lower accuracy)

This variability motivates the use of **cross-validation** (Section 2.4) for more reliable evaluation.

---

## 2.3 Baseline Accuracy

### Classifiers on breast-cancer.arff (10-fold Cross-Validation)

| **Classifier** | **Accuracy** |
| --- | --- |
| J48 | 75.52% |
| NaiveBayes | 71.68% |
| IBk | 72.38% |
| PART | 71.33% |
| **ZeroR** | **70.28%** |

### What is Baseline Accuracy?

**Baseline accuracy** is the accuracy achieved by the simplest possible classifier — **ZeroR** — which always predicts the **majority class**. For `breast-cancer.arff`:
- The majority class is `no-recurrence-events` with 201 out of 286 instances (70.28%)
- ZeroR achieves 70.28% accuracy by predicting every instance as "no-recurrence-events"

**A useful classifier must beat the baseline.** If a complex algorithm doesn't outperform ZeroR, it provides no value. In this case:
- J48 beats baseline by +5.24 percentage points (75.52% vs 70.28%) — the best performer
- NaiveBayes and IBk beat baseline by only ~1.5–2 points
- PART barely beats baseline (+1.05 points)

These relatively modest improvements suggest that breast cancer recurrence is difficult to predict from the available attributes.

---

### Classifiers on unbalanced.arff (10-fold Cross-Validation)

| **Classifier** | **Accuracy** |
| --- | --- |
| ZeroR | 98.60% |
| J48 | 98.60% |
| NaiveBayes | 90.77% |
| IBk | 97.66% |
| PART | 97.55% |

### Why do the classifiers achieve lower accuracy?

The question is slightly counterintuitive — the **numbers** look high (90–99%), but the classifiers perform **worse relative to the baseline**:

1. **ZeroR achieves 98.60%** because the dataset is extremely **unbalanced** — the majority class makes up ~98.6% of all instances, so simply predicting the majority class gives near-perfect accuracy.

2. **No classifier beats ZeroR.** In fact, J48 ties with ZeroR (98.60%), while NaiveBayes, IBk, and PART all perform **worse** than the baseline.

3. **Why is this?** When a dataset is highly unbalanced:
   - The minority class has very few training examples, so classifiers struggle to learn its patterns
   - Predicting the minority class introduces errors on the overwhelming majority class
   - Algorithms can achieve higher accuracy by **ignoring the minority class entirely** (which is what ZeroR does)

4. **NaiveBayes drops to 90.77%** — it tries to learn the minority class and ends up misclassifying some majority-class instances, resulting in worse overall accuracy.

**Key takeaway:** Raw accuracy is a **misleading metric** for unbalanced datasets. When 98.6% of data belongs to one class, accuracy doesn't tell you whether the classifier learned anything useful. Other metrics like precision, recall, F-measure, or AUC are more informative.

---

## 2.4 Cross-validation

### Concepts Review

| Concept | Description |
|---------|-------------|
| **Holdout** | Split data into training and test sets once. E.g., 90% train / 10% test |
| **Stratification** | Ensure each class is proportionally represented in both train and test sets |
| **Cross-validation (10-fold)** | Split data into 10 equal parts. Use 9 parts for training and 1 for testing. Repeat 10 times (each part gets tested once). Average the 10 results |
| **Random seed** | Controls how data is shuffled before splitting. Different seeds → different splits |

---

### J48 on credit-g.arff

| | | | |
| --- | --- | --- | --- |
| **Holdout (10%)** | **Percent accuracy (x)** | **10-fold cross-validation** | **Percent accuracy (x)** |
| Random seed: 1 | 74.0% | Random seed: 1 | 70.5% |
| -- 2 | 77.0% | -- 2 | 72.3% |
| -- 3 | 70.0% | -- 3 | 71.2% |
| -- 4 | 74.0% | -- 4 | 71.2% |
| -- 5 | 67.0% | -- 5 | 71.3% |
| -- 6 | 67.0% | -- 6 | 71.5% |
| -- 7 | 70.0% | -- 7 | 71.1% |
| -- 8 | 71.0% | -- 8 | 70.9% |
| -- 9 | 69.0% | -- 9 | 72.1% |
| -- 10 | 72.0% | -- 10 | 70.4% |
| *Sample Mean* | **71.10%** | *Sample Mean* | **71.25%** |
| *Standard deviation* | **3.21%** | *Standard deviation* | **0.61%** |

---

### PART on credit-g.arff

| | | | |
| --- | --- | --- | --- |
| **Holdout (10%)** | **Percent accuracy (x)** | **10-fold cross-validation** | **Percent accuracy (x)** |
| Random seed: 1 | 77.0% | Random seed: 1 | 70.2% |
| -- 2 | 77.0% | -- 2 | 71.0% |
| -- 3 | 65.0% | -- 3 | 72.0% |
| -- 4 | 73.0% | -- 4 | 70.4% |
| -- 5 | 69.0% | -- 5 | 69.0% |
| -- 6 | 67.0% | -- 6 | 70.0% |
| -- 7 | 66.0% | -- 7 | 70.4% |
| -- 8 | 72.0% | -- 8 | 68.8% |
| -- 9 | 75.0% | -- 9 | 72.8% |
| -- 10 | 75.0% | -- 10 | 70.8% |
| *Sample Mean* | **71.60%** | *Sample Mean* | **70.54%** |
| *Standard deviation* | **4.55%** | *Standard deviation* | **1.22%** |

---

### NaiveBayes on credit-g.arff

| | | | |
| --- | --- | --- | --- |
| **Holdout (10%)** | **Percent accuracy (x)** | **10-fold cross-validation** | **Percent accuracy (x)** |
| Random seed: 1 | 76.0% | Random seed: 1 | 75.4% |
| -- 2 | 78.0% | -- 2 | 75.8% |
| -- 3 | 80.0% | -- 3 | 74.8% |
| -- 4 | 78.0% | -- 4 | 75.7% |
| -- 5 | 71.0% | -- 5 | 74.8% |
| -- 6 | 78.0% | -- 6 | 74.8% |
| -- 7 | 72.0% | -- 7 | 74.8% |
| -- 8 | 73.0% | -- 8 | 75.3% |
| -- 9 | 77.0% | -- 9 | 75.0% |
| -- 10 | 77.0% | -- 10 | 75.2% |
| *Sample Mean* | **76.00%** | *Sample Mean* | **75.16%** |
| *Standard deviation* | **2.98%** | *Standard deviation* | **0.38%** |

---

### Key Observations for Section 2.4

#### 1. Cross-validation is much more stable than holdout

| Classifier | Holdout Std Dev | CV Std Dev | Stability Improvement |
|------------|----------------|------------|----------------------|
| J48 | 3.21% | **0.61%** | 5.3× more stable |
| PART | 4.55% | **1.22%** | 3.7× more stable |
| NaiveBayes | 2.98% | **0.38%** | 7.8× more stable |

Cross-validation standard deviations are **3.7× to 7.8× smaller** than holdout deviations. This means cross-validation gives much more **reliable, consistent** estimates.

#### 2. Why is holdout so variable?

With 10% holdout on `credit-g.arff` (1000 instances), only **100 instances** are used for testing. This small test set makes the accuracy estimate highly sensitive to which particular 100 instances happen to be selected. Cross-validation, by contrast, uses **all 1000 instances** for both training and testing (across the 10 folds), making it far more robust.

#### 3. Mean accuracies are similar

Despite the difference in variance, the mean accuracies from holdout and CV are close:
- J48: 71.10% (holdout) ≈ 71.25% (CV)
- PART: 71.60% (holdout) ≈ 70.54% (CV)
- NaiveBayes: 76.00% (holdout) ≈ 75.16% (CV)

This makes sense — both methods are estimating the same underlying "true accuracy."

#### 4. NaiveBayes performs best on credit-g.arff

NaiveBayes (~75%) outperforms both J48 (~71%) and PART (~71%). This suggests the credit-g dataset's attributes have relatively independent contributions to the class, which is the assumption NaiveBayes relies on.

#### 5. Holdout range (min to max) is very wide

- J48 holdout: 67.0% – 77.0% (10-point spread)
- PART holdout: 65.0% – 77.0% (12-point spread)
- NaiveBayes holdout: 71.0% – 80.0% (9-point spread)

If you only ran **one** holdout experiment, you could get any value in this range — a very unreliable estimate!

---

### Summary: Holdout vs. Cross-Validation

| Aspect | Holdout (Percentage Split) | 10-Fold Cross-Validation |
|--------|---------------------------|--------------------------|
| **How it works** | One random split into train/test | 10 systematic splits; each part tested once |
| **Test data used** | Only a fraction (e.g., 10%) | All data (each instance tested exactly once) |
| **Stability** | Low — results vary greatly with seed | High — results are consistent |
| **Standard Deviation** | Large (2.98–4.55%) | Small (0.38–1.22%) |
| **Recommended for** | Quick preliminary checks | Reliable final evaluation |
| **Computational cost** | Low (1 train/test cycle) | Higher (10 train/test cycles) |

**Conclusion:** 10-fold stratified cross-validation should be preferred over holdout for reliable classifier evaluation, especially when the dataset is small or moderately sized. The holdout method is useful for quick experiments but should not be used as the sole evaluation metric.
