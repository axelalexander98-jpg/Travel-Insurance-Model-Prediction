# Travel Insurance Claim Prediction

A machine learning solution to predict travel insurance claim likelihood, enabling proactive risk management and optimized capital allocation.

## 📋 Project Overview

This project develops a predictive model to identify which travel insurance policyholders are most likely to submit claims. The solution helps insurance companies improve risk assessment, optimize capital reserves, and refine pricing strategies.

### Business Context

Travel insurance companies face unpredictable financial liability from policyholder claims. Without accurate prediction mechanisms, they experience:
- **Inaccurate Capital Reserving**: Under-reserving leads to solvency issues; over-reserving causes inefficient capital utilization
- **Suboptimal Pricing**: High-risk policies may be underpriced relative to their actual expected cost
- **Inefficient Resource Allocation**: Resources wasted investigating low-risk claims instead of focusing on high-probability claims

## 🎯 Project Goals

| Goal Type | Description | Metric |
|-----------|-------------|--------|
| **Primary ML Goal** | Identify high-risk claim policies among all actual claim policies | Model Recall Score |
| **Operational Goal** | Automate high-risk flag creation in underwriting process | Underwriting review time efficiency |
| **Financial Goal** | Reduce overall claim loss ratio (Claims Paid / Premiums Earned) | Decrease in Loss Ratio |
| **Strategic Goal** | Identify key driver features influencing claim probability | Actionable insights reports |

## 📊 Dataset

**Source**: Travel insurance policyholder historical data

**Features** (11 columns):
- `Agency`: Insurance agency name (16 unique values)
- `Agency Type`: Airlines or Travel Agency
- `Distribution Channel`: Online or Offline
- `Product Name`: Insurance product type (26 unique products)
- `Gender`: Customer gender (71.4% missing - dropped)
- `Duration`: Trip duration in days
- `Destination`: Travel destination (138 countries)
- `Net Sales`: Policy sales amount
- `Commission`: Agency commission value
- `Age`: Customer age
- `Claim`: Target variable (Yes/No)

**Dataset Size**: 44,328 records → 43,610 after cleaning

**Class Imbalance**: 
- No Claim: 98.46%
- Claim: 1.54%

## 💰 Cost-Benefit Analysis

| Prediction | Actual | Financial Impact | Business Impact |
|------------|--------|------------------|-----------------|
| **True Negative (TN)** | No Claim → No Claim | +$280 profit | ✅ Ideal - Maximum profitability |
| **False Positive (FP)** | No Claim → Claim | -$50 to -$500 loss | ⚠️ Operational inefficiency |
| **False Negative (FN)** | Claim → No Claim | -$2,329 to -$9,720+ loss | ❌ **CRITICAL** - Threatens solvency |
| **True Positive (TP)** | Claim → Claim | $0 to +$50 profit | ✅ Sustainable - Proper reserves |

**Key Insight**: False Negatives are **5-30x more expensive** than False Positives, making **Recall** the priority metric.

### Business Metrics
- Average Premium (2025): $311 per policy
- Average Claim Payout: $2,609 (6x premium cost)
- Target Profit Margin: 20-50%
- Medical Loss Ratio (MLR): 80-90% of premiums

## 🔧 Technical Implementation

### Data Preprocessing

1. **Data Cleaning**
   - Removed `Gender` column (71.4% missing values)
   - Filtered unrealistic durations (> 4000 days) and ages (> 100 years)
   - Removed negative duration values
   - Retained 43,610 valid records

2. **Feature Engineering**
   - **Numerical Features**: RobustScaler (handles non-parametric distributions)
     - Duration, Net Sales, Commission, Age
   - **Categorical Features**: 
     - OneHotEncoder: Agency Type, Distribution Channel
     - BinaryEncoder: Product Name, Agency, Destination (high cardinality)

3. **Class Imbalance Handling**
   - RandomOverSampler (ROS)
   - SMOTE (Synthetic Minority Over-sampling Technique)

### Model Development

**Evaluation Metric**: F-beta Score where β = √(FN_cost/FP_cost) = √(9720/500) ≈ 4.41

This heavily weights **Recall** over Precision to minimize costly False Negatives.

### Models Tested

| Model | F-beta Mean | F-beta Std | Performance |
|-------|-------------|------------|-------------|
| Decision Tree | 0.074 | 0.012 | Best baseline |
| Bagging | 0.027 | 0.007 | Moderate |
| Random Forest | 0.018 | 0.011 | Low |
| XGBoost | 0.002 | 0.004 | Poor (untuned) |
| KNN | 0.002 | 0.004 | Poor |
| Logistic Regression | 0.000 | 0.000 | Failed |
| AdaBoost | 0.000 | 0.000 | Failed |

## 🏆 Final Model Results

### Decision Tree (Tuned)

**Hyperparameters**:
- `max_depth`: 3
- `min_samples_leaf`: 25
- `min_samples_split`: 23
- `class_weight`: 'balanced'
- `resampling`: RandomOverSampler

**Performance**:
- Train F-beta: 0.470
- Test F-beta: 0.450
- Recall: 63%
- Precision: 7%

**Financial Impact**: Total benefit = -$1,047,500 (vs. -$1,302,480 with no model)
- **Savings: $254,980** (19.6% improvement)

### XGBoost (Tuned) ⭐ **RECOMMENDED**

**Hyperparameters**:
- `n_estimators`: 90
- `max_depth`: 3
- `learning_rate`: 0.0113
- `scale_pos_weight`: 63
- `min_child_weight`: 36
- `max_delta_step`: 1

**Performance**:
- Train F-beta: 0.471
- Test F-beta: 0.466
- Recall: **66%** (catches 89/134 claims)
- Precision: 7%

**Financial Impact**: Total benefit = -$1,041,900 (vs. -$1,302,480 with no model)
- **Savings: $260,580** (20.0% improvement)
- **Additional savings over Decision Tree: $5,600**

### Model Comparison

| Metric | No Model | Decision Tree | XGBoost |
|--------|----------|---------------|---------|
| False Positives | 0 | 1,123 | 1,209 |
| False Negatives | 134 | 50 | 45 |
| Total Loss | -$1,302,480 | -$1,047,500 | -$1,041,900 |
| **Savings** | Baseline | **$254,980** | **$260,580** |

## 📈 Feature Importance (SHAP Analysis)

Top factors influencing claim predictions:

1. **Duration** ⏱️
   - Longer trips = higher claim probability
   - Extended exposure to medical, cancellation risks

2. **Net Sales / Commission** 💵
   - Higher premium products correlate with claims
   - Comprehensive coverage attracts risk-aware customers

3. **Age** 👴
   - Older travelers show higher claim rates
   - Health-related claims increase with age

4. **Destination** 🌍
   - Certain regions have higher claim frequencies

5. **Product Type** 📋
   - Comprehensive plans vs. basic coverage

## 🚀 Installation & Usage

### Requirements

```
scikit-learn==1.7.1
xgboost==2.1.1
imbalanced-learn==0.14.0
numpy==2.3.3
pandas==2.3.2
streamlit==1.49.1
matplotlib==3.10.6
joblib==1.5.2
dill==0.4.0
lime==0.2.0.1
category-encoders==2.8.1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Model Files

- `pipe_tuned_xgboost20251113_1122.pkl` - Production XGBoost model
- `pipe_tuned_dtree20251113_1122.pkl` - Decision Tree model
- `lime_explainer.dill` - LIME explainer for interpretability

### Using the Model

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('pipe_tuned_xgboost20251113_1122.pkl')

# Prepare input data
input_data = pd.DataFrame({
    'Agency': ['EPX'],
    'Agency Type': ['Travel Agency'],
    'Distribution Channel': ['Online'],
    'Product Name': ['Comprehensive Plan'],
    'Duration': [30],
    'Destination': ['SINGAPORE'],
    'Net Sales': [150.0],
    'Commision (in value)': [15.0],
    'Age': [45]
})

# Predict
prediction = model.predict(input_data)
claim_probability = model.predict_proba(input_data)[:, 1]

print(f"Claim Prediction: {'Yes' if prediction[0] else 'No'}")
print(f"Claim Probability: {claim_probability[0]:.2%}")
```

## 💡 Business Recommendations

### 1. Dynamic Premium Pricing
- **High-risk customers**: Adjust premium upward to cover expected costs
- **Low-risk customers**: Competitive pricing to attract profitable segments

### 2. Customer Segmentation
Segment customers by claim likelihood:
- **High Risk** (>50% probability): Manual underwriting review, adjusted premiums
- **Medium Risk** (20-50%): Automated approval with moderate reserves
- **Low Risk** (<20%): Fast-track approval, minimal reserves

### 3. Operational Efficiency
- **Automate low-risk processing**: Reduce underwriting time for 95%+ of applications
- **Focus human review on high-risk**: Allocate expert resources to cases that matter

### 4. Product Development
Design products based on risk drivers:
- Age-specific policies (senior travel insurance)
- Duration-based pricing tiers
- Destination risk categories (high-risk regions)

### 5. Fraud Detection
Use model scores to prioritize claim investigations:
- High predicted probability + actual claim = standard processing
- Low predicted probability + actual claim = potential fraud flag

## 🔮 Future Improvements

### Model Enhancements
1. **Feature Engineering**
   - Group destinations into regions (reduce dimensionality)
   - Add customer historical behavior data
   - Include economic indicators (GDP, healthcare costs by destination)
   - Travel purpose (business vs. leisure)

2. **Encoding Strategy**
   - Use OneHotEncoder where feasible for better interpretability
   - Reduce binary-encoded features by grouping categories

3. **Hyperparameter Tuning**
   - Use GridSearchCV for exhaustive search (currently RandomizedSearchCV)
   - Explore ensemble methods (stacking, voting classifiers)

4. **Threshold Optimization**
   - Tune classification threshold beyond 0.5
   - Balance recall/precision based on business constraints

### Data Collection
- Customer income/occupation data
- Previous insurance history
- Travel frequency patterns
- Pre-existing medical conditions (where legally permissible)
- Booking lead time

## 👥 Stakeholders

| Group | Interest | Model Impact |
|-------|----------|--------------|
| **Risk Management / Actuarial** | Quantifying risk | Data-driven tool for pricing models, reserves, liability estimation |
| **Underwriting** | Approving policies, setting prices | Automated risk scoring, fast-tracking low-risk policies |
| **Finance / Executive** | Capital allocation, profitability | Improved financial forecasting, reduced capital uncertainty |
| **Claims Department** | Payout verification | Proactive resource allocation, fraud detection |
| **Sales / Marketing** | Distribution, customer segmentation | Targeted marketing, product design insights |

## 📝 Model Interpretability

### SHAP (SHapley Additive exPlanations)
- Global feature importance across all predictions
- Individual prediction explanations
- Identifies non-linear relationships

### LIME (Local Interpretable Model-agnostic Explanations)
- Explains individual predictions to customers/regulators
- Justifies premium adjustments
- Ensures regulatory compliance (explainable AI)

### Decision Tree Visualization
- Simple, visual decision rules
- Easy stakeholder communication
- Audit-friendly for compliance

## 📄 License

This project is part of a capstone module for educational purposes.

**Note**: This model is designed to augment human decision-making, not replace it. Always combine model predictions with expert judgment for final underwriting decisions.
