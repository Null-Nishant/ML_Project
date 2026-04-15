# AutoML-Flow: An Integrated Low-Code Platform for End-to-End Machine Learning

AutoML-Flow is a Streamlit dashboard that walks users through an end-to-end ML pipeline with a visually guided, horizontally flowing sequence of steps.

## Features

1. **Problem Type Selection**: Choose Classification or Regression.
2. **Input Data + PCA Visualization**: Upload CSV/XLSX, pick target/features, and inspect PCA-based shape.
3. **EDA**: Missing values, dtypes, summary stats, target distribution, and correlation heatmap.
4. **Data Engineering & Cleaning**:
   - Imputation: mean/median/mode
   - Outlier detection: IQR, Isolation Forest, DBSCAN, OPTICS
   - Optional outlier removal from UI
5. **Feature Selection**:
   - Variance threshold
   - Correlation threshold
   - Information gain (mutual information with target)
6. **Train/Test Split**
7. **Model Selection**:
   - Regression: Linear Regression, SVM, Random Forest, KMeans
   - Classification: Logistic Regression, SVM, Random Forest, KMeans
8. **Model Training + KFold**
9. **Performance Metrics + Overfitting/Underfitting hints**
10. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV with performance comparison.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- KMeans is included as an unsupervised exploratory option in the model step.
- The app uses dark-themed Plotly charts and custom CSS cards for an aesthetic dashboard layout.
# ML_Project
