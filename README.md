यहां कुछ सुधार किए गए हैं ताकि गणना और प्रस्तुति सही और स्पष्ट हो:  

---

## **Naïve Bayes Theorem and Feature Formulas**

### **1. Naïve Bayes Theorem**  
Naïve Bayes एक प्रायिकता-आधारित मॉडल है जो **Bayes' Theorem** पर निर्भर करता है और फीचर्स को स्वतंत्र मानता है।  

\[
P(Y | X) = \frac{P(X | Y) P(Y)}{P(X)}
\]

जहाँ:  
- \(P(Y | X)\) = Posterior Probability (X दिए जाने पर Y की प्रायिकता)  
- \(P(X | Y)\) = Likelihood (Y दिए जाने पर X की प्रायिकता)  
- \(P(Y)\) = Prior Probability (किसी क्लास Y की पूर्व प्रायिकता)  
- \(P(X)\) = Evidence (X की कुल प्रायिकता)  

क्योंकि \(P(X)\) सभी क्लास के लिए समान है:

\[
P(Y | X) \propto P(X | Y) P(Y)
\]

### **2. Naïve Bayes Assumption**  
Naïve Bayes यह मानता है कि प्रत्येक फीचर क्लास \(Y\) के लिए स्वतंत्र है:

\[
P(X | Y) = P(x_1 | Y) P(x_2 | Y) ... P(x_n | Y)
\]

अतः:

\[
P(Y | X) \propto P(Y) \prod_{i=1}^{n} P(x_i | Y)
\]

---

## **3. Types of Naïve Bayes Classifiers**

### **A. Gaussian Naïve Bayes (Continuous Features)**
यदि फीचर्स **Gaussian (Normal) Distribution** को फॉलो करते हैं, तो:

\[
P(x_i | Y) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2 \sigma^2}}
\]

जहाँ:  
- \( \mu \) = Mean (औसत)  
- \( \sigma \) = Standard Deviation (मानक विचलन)  

---

### **B. Multinomial Naïve Bayes (Categorical Features)**
यह तब उपयोगी होता है जब फीचर्स शब्दों की फ्रीक्वेंसी दर्शाते हैं:

\[
P(x_i | Y) = \frac{count(x_i, Y) + 1}{\sum_{j} count(x_j, Y) + |V|}
\]

जहाँ:  
- \( count(x_i, Y) \) = क्लास \(Y\) में फीचर \(x_i\) की फ्रीक्वेंसी  
- \( |V| \) = कुल शब्दावली की संख्या  

---

### **C. Bernoulli Naïve Bayes (Binary Features)**
जब फीचर्स बाइनरी होते हैं (0 या 1), तो:

\[
P(x_i | Y) =
\begin{cases}
P_i, & \text{if } x_i = 1 \\
1 - P_i, & \text{if } x_i = 0
\end{cases}
\]

जहाँ \( P_i \) = क्लास \(Y\) में \( x_i \) के उपस्थित होने की प्रायिकता।  

---

## **4. Class Prediction**
\[
P(Y_k | X) = P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
\]

सबसे अधिक स्कोर वाली क्लास चुनी जाती है:

\[
Y_{pred} = \arg\max_{Y_k} P(Y_k) \prod_{i=1}^{n} P(x_i | Y_k)
\]

---

# **Naïve Bayes Classifier: Hand Calculation Example**

### **Step 1: Given Dataset**  
| Person | Height (cm) | Weight (kg) | Sport Type  |
|--------|------------|------------|-------------|
| A      | 180        | 80         | Basketball  |
| B      | 175        | 75         | Basketball  |
| C      | 190        | 85         | Basketball  |
| D      | 160        | 55         | Tennis      |
| E      | 165        | 60         | Tennis      |
| F      | 155        | 50         | Tennis      |

**Goal:**  
Predict **Sport Type** for a new person with:  
- **Height = 170 cm**  
- **Weight = 65 kg**  

---

### **Step 2: Compute Prior Probabilities**
\[
P(Basketball) = \frac{3}{6} = 0.5
\]
\[
P(Tennis) = \frac{3}{6} = 0.5
\]

---

### **Step 3: Compute Mean and Standard Deviation**  
#### **For Basketball Players**
\[
\mu_{height} = \frac{180 + 175 + 190}{3} = 181.67
\]
\[
\sigma_{height} = \sqrt{\frac{(180-181.67)^2 + (175-181.67)^2 + (190-181.67)^2}{2}} = 7.57
\]
\[
\mu_{weight} = \frac{80 + 75 + 85}{3} = 80
\]
\[
\sigma_{weight} = 5
\]

#### **For Tennis Players**
\[
\mu_{height} = \frac{160 + 165 + 155}{3} = 160
\]
\[
\sigma_{height} = 5
\]
\[
\mu_{weight} = \frac{55 + 60 + 50}{3} = 55
\]
\[
\sigma_{weight} = 5
\]

---

### **Step 4: Compute Likelihood Probabilities**
For **Height = 170 cm, Weight = 65 kg**, using Gaussian formula:

#### **Basketball Probability**
\[
P(170 | Basketball) = \frac{1}{\sqrt{2\pi (7.57)^2}} e^{-\frac{(170 - 181.67)^2}{2 (7.57)^2}} = 0.0213
\]
\[
P(65 | Basketball) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(65 - 80)^2}{2 (5)^2}} = 0.0007
\]

#### **Tennis Probability**
\[
P(170 | Tennis) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(170 - 160)^2}{2 (5)^2}} = 0.0024
\]
\[
P(65 | Tennis) = \frac{1}{\sqrt{2\pi (5)^2}} e^{-\frac{(65 - 55)^2}{2 (5)^2}} = 0.0024
\]

---

### **Step 5: Compute Posterior Probabilities**
\[
P(Basketball | X) = P(170 | Basketball) P(65 | Basketball) P(Basketball)
\]
\[
= (0.0213) (0.0007) (0.5) = 7.46 \times 10^{-6}
\]

\[
P(Tennis | X) = P(170 | Tennis) P(65 | Tennis) P(Tennis)
\]
\[
= (0.0024) (0.0024) (0.5) = 2.88 \times 10^{-6}
\]

---

### **Step 6: Prediction**
\[
P(Basketball | X) > P(Tennis | X)
\]

**Final Prediction:**  
The person is most likely a **Basketball player**. ✅  

---

## **Conclusion**
✅ **Naïve Bayes theorem** का उपयोग किया गया।  
✅ **Gaussian Probability Distribution** का उपयोग हुआ।  
✅ **Prediction** Basketball के पक्ष में गया।  

इस तरीके से, **Naïve Bayes Classifier** का उपयोग वास्तविक डेटा पर किया जा सकता है। 🚀
