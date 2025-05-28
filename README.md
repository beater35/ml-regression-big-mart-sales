# Big Mart Sales Prediction

A machine learning project that predicts item outlet sales for Big Mart stores using various product and outlet characteristics.

## 📊 Dataset

The dataset contains 8,523 samples with 11 features and 1 target variable:

### Features:
1. **Item_Identifier** - Unique product ID
2. **Item_Weight** - Weight of product
3. **Item_Fat_Content** - Whether the product is low fat or regular
4. **Item_Visibility** - The percentage of total display area allocated to the product
5. **Item_Type** - The category to which the product belongs
6. **Item_MRP** - Maximum Retail Price (list price) of the product
7. **Outlet_Identifier** - Unique store ID
8. **Outlet_Establishment_Year** - The year in which store was established
9. **Outlet_Size** - The size of the store (Small/Medium/High)
10. **Outlet_Location_Type** - The type of area where the store is located
11. **Outlet_Type** - Whether the outlet is a grocery store or supermarket

### Target Variable:
- **Item_Outlet_Sales**: Sales of the product in the particular store

## 🚀 Models Implemented

### 1. XGBoost Regressor (Primary Model)
- **Test R² Score**: 58.6% (optimized)
- **Training R² Score**: 62.2% (optimized)
- **Cross-validation Score**: 59.5% (±0.013)
- **Optimized with GridSearchCV**

### 2. Linear Regression (Baseline Model)
- **Test R² Score**: 48.9%
- **Training R² Score**: 50.7%
- **Simple baseline for comparison**

## 🔧 Key Features

- **Data Preprocessing**: 
  - Missing value imputation (Item_Weight with mean, Outlet_Size with mode by outlet type)
  - Categorical data standardization
  - Label encoding for categorical variables
- **Feature Engineering**: Top feature selection based on importance/coefficients
- **Hyperparameter Optimization**: GridSearchCV for model tuning
- **Cross-validation**: 5-fold CV for robust evaluation

## 📈 Top Features by Importance

### XGBoost Model:
1. **Outlet_Type** (41.5% importance)
2. **Item_MRP** (23.2% importance)
3. **Outlet_Establishment_Year** (17.2% importance)
4. **Outlet_Identifier** (5.0% importance)

### Linear Regression Model:
1. **Item_Visibility** (highest coefficient magnitude)
2. **Outlet_Type**
3. **Outlet_Size**
4. **Outlet_Location_Type**

## 🛠️ Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Running the Code
```python
# Clone the repository
git clone https://github.com/beater35/ml-regression-big-mart-sales.git
cd ml-regression-big-mart-sales

# Load and run the Jupyter notebook
jupyter notebook regression_model_comparison_big_mart_sales.ipynb
```

### Making Predictions
```python
# Example prediction with XGBoost (top 4 features)
new_data = {
    'Outlet_Type': 1,
    'Item_MRP': 249.8092,
    'Outlet_Establishment_Year': 1999,
    'Outlet_Identifier': 9
}

prediction = final_xgb_model.predict([list(new_data.values())])
print(f"Predicted Sales: ${prediction[0]:.2f}")
```

## 📊 Model Performance Summary

| Model | Test R² Score | Training R² Score | Cross-Val Score |
|-------|---------------|-------------------|-----------------|
| **XGBoost (Optimized)** | **58.6%** | **62.2%** | **59.5% ± 1.3%** |
| Linear Regression | 48.9% | 50.7% | - |

## 🔍 Project Structure

```
bigmart-sales-prediction/
├── regression_model_comparison_big_mart_sales.ipynb
└── README.md
```

## 📝 Data Preprocessing Steps

1. **Missing Value Treatment**:
   - Item_Weight: Filled with mean value (12.86)
   - Outlet_Size: Filled with mode based on outlet type

2. **Data Standardization**:
   - Item_Fat_Content: Standardized variations (LF→Low Fat, reg→Regular)

3. **Label Encoding**: All categorical variables converted to numerical format

4. **Feature Selection**: Top N features selected based on model importance/coefficients

## 📈 Best Hyperparameters (XGBoost)

```python
{
    'colsample_bytree': 0.9,
    'learning_rate': 0.1,
    'max_depth': 3,
    'n_estimators': 100,
    'subsample': 0.8
}
```

## 🎯 Results & Insights

- **XGBoost significantly outperforms Linear Regression** with 10% higher R² score
- **Outlet_Type is the most important predictor** (41.5% importance)
- **Item_MRP (pricing) is crucial** for sales prediction (23.2% importance)
- **Store characteristics matter more than product characteristics** for sales prediction
- **Model achieves reasonable performance** with R² = 0.586 on test data

## 📊 Visualizations Included

- Distribution plots for numerical features
- Count plots for categorical features
- Feature importance rankings
- Model performance comparisons

## 📚 References

- Dataset: [Kaggle - Big Mart Sales Dataset](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)
