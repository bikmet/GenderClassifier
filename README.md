# Gender Classification from Physical Attributes: Model Evaluation & Analysis

A comparative classification experiment evaluating three foundational models—**Decision Tree**, **Logistic Regression**, and **Gradient Boosting**—on a biometric dataset (Height, Weight, Shoe Size) to determine gender.

---

## Experimental Setup

* **Total Dataset Size:** 40 samples (20 Male, 20 Female)
* **Features:**
  1. Height ($cm$)
  2. Weight ($kg$)
  3. Shoe Size ($EU$)
 
---

## Evaluation Results

| Model | Evaluation Accuracy | Correct Predictions | Errors |
| :--- | :---: | :---: | :---: |
| **DecisionTreeClassifier** | **83.33%** | 10 / 12 | 2 |
| **GradientBoostingClassifier** | **83.33%** | 10 / 12 | 2 |
| **LogisticRegression** | **66.67%** | 8 / 12 | 4 |

---

## Result Justification & Technical Analysis

The entire 16.66% gap between the tree-based models and Logistic Regression is driven by **exactly two misclassified samples** (2 / 12). The root causes stem from:

### 1. Tree Models: Scale Invariance & Orthogonal Splits (10/12)
* Decision Trees and Gradient Boosted Trees do not compute vector dot products or distances. They search for individual threshold cuts per feature.
* Because physical attributes naturally correlate with gender, axis-aligned cuts easily isolate the majority of class clusters in low-dimensional space.
* The 2 misclassified instances correspond to overlapping border samples (e.g., taller females with larger builds, such as `[175, 72, 41]` and `[177, 74, 42]`), which fall across standard gender threshold boundaries.

### 2. Logistic Regression: Sensitivity to Scale & Regularization (8/12)
Unlike trees, Logistic Regression models a global linear decision boundary:

$$P(Y = \text{male} \mid X) = \sigma(w_1 \cdot \text{Height} + w_2 \cdot \text{Weight} + w_3 \cdot \text{Shoe Size} + b)$$

Without feature scaling, performance degrades due to two mechanics:
1. **Unbalanced Feature Magnitudes:** Height values (~155–192) are roughly $4\times$ to $5\times$ larger than Shoe Size values (~36–47). During gradient optimization, larger features dominate weight updates and loss gradients, causing the model to rely disproportionately on Height while muting the relative predictive contribution of Shoe Size.
2. **Hyperplane Distortion in Small Samples:** With only 28 training rows, the unscaled optimization landscape causes the linear hyperplane to misclassify 2 additional ambiguous border samples near the transition boundary
