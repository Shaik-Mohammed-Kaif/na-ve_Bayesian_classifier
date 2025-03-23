### **Naïve Bayes Theorem and All Feature Formulas**  

#### **1. Naïve Bayes Theorem**  
The Naïve Bayes theorem is a probabilistic classifier based on **Bayes’ theorem** with the assumption that features are independent.  

\[
P(Y | X) = \frac{P(X | Y) P(Y)}{P(X)}
\]

Where:  
- \( P(Y | X) \) → **Posterior Probability** (Probability of class \( Y \) given feature vector \( X \))  
- \( P(X | Y) \) → **Likelihood** (Probability of feature vector \( X \) given class \( Y \))  
- \( P(Y) \) → **Prior Probability** (Probability of class \( Y \) before observing features)  
- \( P(X) \) → **Evidence** (Probability of feature vector \( X \) across all classes)

Since \( P(X) \) is constant for all classes, we use:

\[
P(Y | X) \propto P(X | Y) P(Y)
\]

---

#### **2. Assumption of Independence (Naïve Bayes Assumption)**  
The Naïve Bayes classifier assumes **each feature is conditionally independent** given the class label \( Y \):

\[
P(X | Y) = P(x_1, x_2, ..., x_n | Y) = P(x_1 | Y) P(x_2 | Y) ... P(x_n | Y)
\]

Thus, the formula simplifies to:

\[
P(Y | X) \propto P(Y) \prod_{i=1}^{n} P(x_i | Y)
\]

---

#### **3. Types of Naïve Bayes Classifiers and Feature Probability Calculation**
##### **A. Gaussian Naïve Bayes (For Continuous Features)**
If a feature follows a **normal distribution**, we use the **Gaussian (Normal) Distribution**:

\[
P(x_i | Y) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{-(x_i - \mu)^2}{2 \sigma^2}}
\]

Where:  
- \( \mu \) = Mean of feature \( x_i \) for class \( Y \)  
- \( \sigma \) = Standard deviation of feature \( x_i \) for class \( Y \)  

##### **B. Multinomial Naïve Bayes (For Categorical/Discrete Features)**
Used when features represent frequency counts (e.g., word occurrences in text classification). The probability is:

\[
P(x_i | Y) = \frac{count(x_i, Y) + 1}{\sum_{j} count(x_j, Y) + |V|}
\]

Where:  
- \( count(x_i, Y) \) → Number of times feature \( x_i \) appears in class \( Y \)  
- \( |V| \) → Total number of unique features (Vocabulary size)  
- **Laplace Smoothing** (+1) prevents zero probabilities

##### **C. Bernoulli Naïve Bayes (For Binary Features)**
Used for binary features (0 or 1), such as presence/absence of a word in text classification:

\[
P(x_i | Y) =
\begin{cases}
P_i & \text{if } x_i = 1 \\
1 - P_i & \text{if } x_i = 0
\end{cases}
\]

Where \( P_i \) is the probability of \( x_i = 1 \) in class \( Y \).

---

#### **4. Class Prediction**
For a given test feature vector \( X = (x_1, x_2, ..., x_n) \), compute:

\[
P(Y_k | X) = P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
\]

The predicted class \( Y_{pred} \) is:

\[
Y_{pred} = \arg\max_{Y_k} P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
\]

This selects the class \( Y_k \) with the highest probability.

---

### **Final Summary**
1. **Bayes’ Theorem** is applied to classification.
2. **Naïve Bayes Assumption** considers independent features.
3. **Gaussian Naïve Bayes** handles continuous data using a normal distribution.
4. **Multinomial Naïve Bayes** is used for text classification (word frequency).
5. **Bernoulli Naïve Bayes** is used for binary features (word presence/absence).
6. **Prediction** is made by selecting the class with the highest probability.

This method is computationally efficient and works well with large datasets, especially in text classification and spam detection.
