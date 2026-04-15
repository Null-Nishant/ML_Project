import io
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AutoML-Flow", page_icon="🤖", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.1rem;
            background: linear-gradient(90deg, #6366f1, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-title {
            color: #cbd5e1;
            margin-bottom: 1.4rem;
        }
        .flow-wrap {
            display: flex;
            overflow-x: auto;
            gap: 0.6rem;
            padding-bottom: 0.4rem;
            margin-bottom: 1rem;
        }
        .flow-step {
            min-width: 160px;
            border-radius: 14px;
            padding: 0.7rem 0.8rem;
            background: #1f2937;
            border: 1px solid #374151;
            color: #f8fafc;
            text-align: center;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .flow-step-active {
            background: linear-gradient(90deg, #2563eb, #06b6d4);
            border: none;
        }
        .card {
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 0.8rem;
            background: #0f172a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@dataclass
class ModelBundle:
    model_name: str
    estimator: object
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def horizontal_flow(active_step: int) -> None:
    steps = [
        "1. Problem",
        "2. Input + PCA",
        "3. EDA",
        "4. Cleaning",
        "5. Feature Select",
        "6. Split",
        "7. Model Select",
        "8. Train + KFold",
        "9. Metrics",
        "10. Tuning",
    ]
    html = "<div class='flow-wrap'>"
    for i, step in enumerate(steps, 1):
        cls = "flow-step flow-step-active" if i == active_step else "flow-step"
        html += f"<div class='{cls}'>{step}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def parse_uploaded_data(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        st.error("Unsupported file type. Upload CSV or Excel.")
        return None
    except Exception as e:
        st.error(f"Error parsing uploaded file: {str(e)}")
        st.info("Suggestion: Ensure the file is not corrupted, check file format, and verify column headers are present.")
        return None


def run_pca_visual(df: pd.DataFrame, selected_features: List[str]) -> None:
    if len(selected_features) < 2:
        st.info("Select at least 2 numerical features for PCA view.")
        return
    subset = df[selected_features].dropna()
    if subset.empty:
        st.warning("No rows available after removing missing values for PCA.")
        return

    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        title="PCA Projection (2D)",
        opacity=0.8,
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)
    explained = pca.explained_variance_ratio_ * 100
    st.caption(f"Explained variance: PC1={explained[0]:.2f}% | PC2={explained[1]:.2f}%")


def quick_eda(df: pd.DataFrame, target_col: str, problem_type: str) -> None:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Missing Values")
        miss = df.isna().sum().sort_values(ascending=False)
        miss_df = miss[miss > 0].reset_index()
        miss_df.columns = ["Column", "Missing Count"]
        st.dataframe(miss_df if not miss_df.empty else pd.DataFrame({"Info": ["No missing values"]}))

    with c2:
        st.markdown("#### Data Types")
        dtype_df = df.dtypes.astype(str).reset_index()
        dtype_df.columns = ["Column", "Dtype"]
        st.dataframe(dtype_df)

    st.markdown("#### Summary Statistics")
    st.dataframe(df.describe(include="all").transpose())

    num_cols, cat_cols = detect_column_types(df)
    if num_cols:
        st.markdown("#### Correlation Heatmap (Numerical)")
        corr = df[num_cols].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Teal", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if target_col in df.columns:
        st.markdown("#### Target Distribution")
        if problem_type == "Classification":
            fig = px.histogram(df, x=target_col, template="plotly_dark", color=target_col)
        else:
            fig = px.histogram(df, x=target_col, template="plotly_dark", nbins=40)
        st.plotly_chart(fig, use_container_width=True)


def apply_data_cleaning(df: pd.DataFrame, numeric_cols: List[str], impute_method: str) -> pd.DataFrame:
    cleaned = df.copy()
    strategy = {"Mean": "mean", "Median": "median", "Mode": "most_frequent"}[impute_method]
    if numeric_cols:
        imp = SimpleImputer(strategy=strategy)
        cleaned[numeric_cols] = imp.fit_transform(cleaned[numeric_cols])
    cat_cols = [c for c in cleaned.columns if c not in numeric_cols]
    if cat_cols:
        imp_c = SimpleImputer(strategy="most_frequent")
        cleaned[cat_cols] = imp_c.fit_transform(cleaned[cat_cols])
    return cleaned


def detect_outliers(df: pd.DataFrame, method: str, numeric_cols: List[str]) -> pd.Series:
    if not numeric_cols:
        return pd.Series(False, index=df.index)

    x = df[numeric_cols].copy()
    x = x.fillna(x.median(numeric_only=True))

    if method == "IQR":
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        mask = ((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))).any(axis=1)
        return mask
    if method == "Isolation Forest":
        iso = IsolationForest(contamination=0.05, random_state=42)
        pred = iso.fit_predict(x)
        return pd.Series(pred == -1, index=df.index)
    if method == "DBSCAN":
        scaled = StandardScaler().fit_transform(x)
        db = DBSCAN(eps=1.4, min_samples=8)
        labels = db.fit_predict(scaled)
        return pd.Series(labels == -1, index=df.index)
    if method == "OPTICS":
        scaled = StandardScaler().fit_transform(x)
        op = OPTICS(min_samples=10)
        labels = op.fit_predict(scaled)
        return pd.Series(labels == -1, index=df.index)

    return pd.Series(False, index=df.index)


def perform_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    variance_threshold: float,
    corr_threshold: float,
    top_k_info_gain: int,
) -> List[str]:
    selected = X.columns.tolist()

    # Variance threshold on numerical columns only
    num_cols = X[selected].select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        vt = VarianceThreshold(threshold=variance_threshold)
        vt.fit(X[num_cols].fillna(0))
        keep_num = [c for c, keep in zip(num_cols, vt.get_support()) if keep]
        selected = [c for c in selected if c in keep_num or c not in num_cols]

    # Correlation removal among numerical features
    num_selected = X[selected].select_dtypes(include=np.number).columns.tolist()
    if len(num_selected) > 1:
        corr_matrix = X[num_selected].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
        selected = [c for c in selected if c not in to_drop]

    # Information gain only with target (numerical only for simplicity)
    num_selected = X[selected].select_dtypes(include=np.number).columns.tolist()
    if num_selected:
        X_num = X[num_selected].fillna(0)
        if problem_type == "Classification":
            mi = mutual_info_classif(X_num, y)
        else:
            mi = mutual_info_regression(X_num, y)
        mi_df = pd.DataFrame({"feature": num_selected, "mi": mi}).sort_values("mi", ascending=False)
        top_features = mi_df.head(top_k_info_gain)["feature"].tolist()
        selected = [c for c in selected if c in top_features or c not in num_selected]

    return selected


def get_model_and_params(problem_type: str, model_name: str, svm_kernel: str):
    if problem_type == "Regression":
        if model_name == "Linear Regression":
            return LinearRegression(), {
                "model__fit_intercept": [True, False],
                "model__positive": [False, True],
            }
        if model_name == "SVM":
            return SVR(kernel=svm_kernel), {
                "model__C": [0.1, 1, 10],
                "model__epsilon": [0.01, 0.1, 0.2],
            }
        if model_name == "Random Forest":
            return RandomForestRegressor(random_state=42), {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
            }
    else:
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000), {
                "model__C": [0.1, 1, 10],
                "model__solver": ["lbfgs", "liblinear"],
            }
        if model_name == "SVM":
            return SVC(kernel=svm_kernel, probability=True), {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", "auto"],
            }
        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=42), {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
            }
    return None, {}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols, cat_cols = detect_column_types(X)
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, num_cols), ("cat", categorical_transformer, cat_cols)]
    )


def evaluate(problem_type: str, y_true, y_pred, dataset_name: str) -> Dict[str, float]:
    if problem_type == "Regression":
        return {
            f"{dataset_name} R2": r2_score(y_true, y_pred),
            f"{dataset_name} RMSE": mean_squared_error(y_true, y_pred, squared=False),
            f"{dataset_name} MAE": mean_absolute_error(y_true, y_pred),
        }
    return {
        f"{dataset_name} Accuracy": accuracy_score(y_true, y_pred),
        f"{dataset_name} Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{dataset_name} Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        f"{dataset_name} F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def overfit_underfit_hint(problem_type: str, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> str:
    if problem_type == "Regression":
        gap = train_metrics["Train R2"] - test_metrics["Test R2"]
        if gap > 0.15:
            return "⚠️ Potential overfitting (Train R2 significantly higher than Test R2)."
        if test_metrics["Test R2"] < 0.3:
            return "⚠️ Potential underfitting (both train and test performance are low)."
        return "✅ Model fit looks balanced."
    gap = train_metrics["Train Accuracy"] - test_metrics["Test Accuracy"]
    if gap > 0.10:
        return "⚠️ Potential overfitting (Train Accuracy much higher than Test Accuracy)."
    if test_metrics["Test Accuracy"] < 0.6:
        return "⚠️ Potential underfitting (test score is relatively low)."
    return "✅ Model fit looks balanced."


def run_kmeans_panel(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    st.markdown("### KMeans (Unsupervised option)")
    if len(numeric_cols) < 2:
        st.info("KMeans requires at least 2 numerical features.")
        return

    k = st.slider("Select number of clusters (K)", 2, 12, 3)
    km_features = st.multiselect("Select features for clustering", options=numeric_cols, default=numeric_cols[: min(4, len(numeric_cols))])
    if len(km_features) < 2:
        st.warning("Select at least 2 features for KMeans.")
        return

    x = df[km_features].dropna()
    x_scaled = StandardScaler().fit_transform(x)
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = model.fit_predict(x_scaled)

    pca = PCA(n_components=2, random_state=42)
    p = pca.fit_transform(x_scaled)
    plot_df = pd.DataFrame({"PC1": p[:, 0], "PC2": p[:, 1], "Cluster": labels.astype(str)})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", template="plotly_dark", title="KMeans Cluster View (PCA)")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Inertia", f"{model.inertia_:.3f}")


# ---------- Header ----------
st.markdown('<div class="main-title">AutoML-Flow</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An Integrated Low-Code Platform for End-to-End Machine Learning</div>',
    unsafe_allow_html=True,
)

horizontal_flow(active_step=1)

# ---------- Step 1: Problem Type ----------
problem_type = st.radio("1) Select Problem Type", ["Classification", "Regression"], horizontal=True)

# ---------- Step 2: Input Data ----------
horizontal_flow(active_step=2)
uploaded = st.file_uploader("2) Upload your data (CSV/XLSX)", type=["csv", "xlsx", "xls"])

if uploaded:
    df = parse_uploaded_data(uploaded)
    if df is not None:
        st.success(f"Loaded dataset with shape: {df.shape}")
        st.dataframe(df.head(20), use_container_width=True)

        target_col = st.selectbox("Select target column", options=df.columns.tolist())
        feature_options = [c for c in df.columns if c != target_col]
        selected_features = st.multiselect("Select input features", options=feature_options, default=feature_options)

        filtered_df = df[selected_features + [target_col]].copy() if selected_features else df[[target_col]].copy()
        st.info(f"Current data shape after feature selection: {filtered_df.shape}")

        numeric_for_pca = filtered_df[selected_features].select_dtypes(include=np.number).columns.tolist() if selected_features else []
        run_pca_visual(filtered_df, numeric_for_pca)

        # ---------- Step 3: EDA ----------
        horizontal_flow(active_step=3)
        with st.expander("3) Exploratory Data Analysis (EDA)", expanded=True):
            quick_eda(filtered_df, target_col, problem_type)

        # ---------- Step 4: Data Engineering & Cleaning ----------
        horizontal_flow(active_step=4)
        with st.expander("4) Data Engineering & Cleaning", expanded=True):
            numeric_cols, _ = detect_column_types(filtered_df)
            impute_method = st.selectbox("Missing value imputation", ["Mean", "Median", "Mode"])
            try:
                cleaned_df = apply_data_cleaning(filtered_df, numeric_cols, impute_method)
            except Exception as e:
                st.error(f"Error during data cleaning: {str(e)}")
                st.info("Suggestion: Check for data types, ensure numerical columns are properly formatted, and verify imputation method compatibility.")
                st.stop()

            outlier_method = st.selectbox("Outlier detection method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
            try:
                outlier_mask = detect_outliers(cleaned_df.drop(columns=[target_col]), outlier_method, [c for c in numeric_cols if c != target_col])
            except Exception as e:
                st.error(f"Error detecting outliers: {str(e)}")
                st.info("Suggestion: Ensure there are numerical columns, check for infinite values, and try a different outlier detection method.")
                outlier_mask = pd.Series(False, index=cleaned_df.index)

            outlier_count = int(outlier_mask.sum())
            st.write(f"Detected outliers: **{outlier_count}**")
            if outlier_count > 0:
                remove_outliers = st.radio("Do you want to remove detected outliers?", ["No", "Yes"], horizontal=True)
                if remove_outliers == "Yes":
                    cleaned_df = cleaned_df.loc[~outlier_mask].reset_index(drop=True)
                    st.success(f"Outliers removed. New shape: {cleaned_df.shape}")
                else:
                    st.info("Outliers retained.")
            else:
                st.success("No outliers detected by selected method.")

            st.dataframe(cleaned_df.head(20), use_container_width=True)

        # ---------- Step 5: Feature Selection ----------
        horizontal_flow(active_step=5)
        with st.expander("5) Feature Selection", expanded=True):
            X = cleaned_df.drop(columns=[target_col])
            y = cleaned_df[target_col]

            variance_threshold = st.slider("Variance threshold", 0.0, 0.3, 0.0, 0.01)
            corr_threshold = st.slider("Correlation threshold", 0.5, 0.99, 0.9, 0.01)
            top_k_info = st.slider("Top K numerical features by Information Gain", 1, max(1, X.select_dtypes(include=np.number).shape[1]), min(5, max(1, X.select_dtypes(include=np.number).shape[1])))

            try:
                selected = perform_feature_selection(X, y, problem_type, variance_threshold, corr_threshold, top_k_info)
                X_selected = X[selected]
                st.write(f"Selected {len(selected)} features:")
                st.code(", ".join(selected) if selected else "No features selected")
            except Exception as e:
                st.error(f"Error during feature selection: {str(e)}")
                st.info("Suggestion: Ensure target column has variance, check for constant features, and verify numerical columns exist for information gain.")
                selected = X.columns.tolist()
                X_selected = X[selected]

        # ---------- Step 6: Data Split ----------
        horizontal_flow(active_step=6)
        with st.expander("6) Train/Test Split", expanded=True):
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42)

            try:
                stratify_opt = y if (problem_type == "Classification" and y.nunique() > 1) else None
                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify_opt,
                )
                st.write(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
            except Exception as e:
                st.error(f"Error during train/test split: {str(e)}")
                st.info("Suggestion: Ensure sufficient data points for the test size, check for class imbalance in classification, and verify selected features exist.")
                st.stop()

        # ---------- Step 7: Model Selection ----------
        horizontal_flow(active_step=7)
        with st.expander("7) Model Selection", expanded=True):
            if problem_type == "Regression":
                model_choices = ["Linear Regression", "SVM", "Random Forest", "KMeans"]
            else:
                model_choices = ["Logistic Regression", "SVM", "Random Forest", "KMeans"]

            chosen_model = st.selectbox("Choose model", model_choices)
            svm_kernel = st.selectbox("SVM kernel", ["linear", "rbf", "poly", "sigmoid"], disabled=chosen_model != "SVM")

            if chosen_model == "KMeans":
                run_kmeans_panel(cleaned_df.drop(columns=[target_col]), X_selected.select_dtypes(include=np.number).columns.tolist())
                st.stop()

            try:
                estimator, param_grid = get_model_and_params(problem_type, chosen_model, svm_kernel)
                preprocessor = build_preprocessor(X_selected)
                pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
                bundle = ModelBundle(chosen_model, estimator, pipeline, X_train, X_test, y_train, y_test)
            except Exception as e:
                st.error(f"Error building model pipeline: {str(e)}")
                st.info("Suggestion: Check model compatibility with data types, ensure sufficient features, and verify target variable format.")
                st.stop()

        # ---------- Step 8: Training + KFold ----------
        horizontal_flow(active_step=8)
        with st.expander("8) Training + KFold Validation", expanded=True):
            k_value = st.slider("Choose K for KFold", 3, 10, 5)
            scoring = "r2" if problem_type == "Regression" else "accuracy"

            try:
                bundle.pipeline.fit(bundle.X_train, bundle.y_train)
                cv = KFold(n_splits=k_value, shuffle=True, random_state=42)
                cv_scores = cross_val_score(bundle.pipeline, bundle.X_train, bundle.y_train, cv=cv, scoring=scoring)

                st.write(f"KFold ({k_value}) mean {scoring}: **{cv_scores.mean():.4f}**")
                st.line_chart(pd.DataFrame({"Fold": np.arange(1, k_value + 1), "Score": cv_scores}).set_index("Fold"))
            except Exception as e:
                st.error(f"Error during training or cross-validation: {str(e)}")
                st.info("Suggestion: Check for data issues like NaN values, ensure sufficient data for K folds, and verify model parameters.")
                st.stop()

        # ---------- Step 9: Metrics + Over/Underfit ----------
        horizontal_flow(active_step=9)
        with st.expander("9) Performance Metrics + Fit Check", expanded=True):
            try:
                train_pred = bundle.pipeline.predict(bundle.X_train)
                test_pred = bundle.pipeline.predict(bundle.X_test)

                train_metrics = evaluate(problem_type, bundle.y_train, train_pred, "Train")
                test_metrics = evaluate(problem_type, bundle.y_test, test_pred, "Test")

                metrics_df = pd.DataFrame([train_metrics, test_metrics]).T
                metrics_df.columns = ["Train", "Test"]
                st.dataframe(metrics_df, use_container_width=True)

                hint = overfit_underfit_hint(problem_type, train_metrics, test_metrics)
                st.markdown(f"### {hint}")

                bar_df = pd.DataFrame(
                    {
                        "Metric": list(train_metrics.keys()) + list(test_metrics.keys()),
                        "Value": list(train_metrics.values()) + list(test_metrics.values()),
                        "Set": ["Train"] * len(train_metrics) + ["Test"] * len(test_metrics),
                    }
                )
                fig = px.bar(bar_df, x="Metric", y="Value", color="Set", barmode="group", template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error during evaluation: {str(e)}")
                st.info("Suggestion: Ensure model is trained, check for prediction errors, and verify target variable consistency.")
                st.stop()

        # ---------- Step 10: Hyperparameter Tuning ----------
        horizontal_flow(active_step=10)
        with st.expander("10) Hyperparameter Tuning", expanded=True):
            tuning_method = st.radio("Select tuning method", ["GridSearch", "RandomSearch"], horizontal=True)
            tune_runs = st.slider("RandomSearch iterations", 5, 30, 10, disabled=tuning_method != "RandomSearch")

            if st.button("Run Hyperparameter Tuning"):
                try:
                    if tuning_method == "GridSearch":
                        search = GridSearchCV(bundle.pipeline, param_grid=param_grid, cv=5, n_jobs=-1, scoring=scoring)
                    else:
                        search = RandomizedSearchCV(
                            bundle.pipeline,
                            param_distributions=param_grid,
                            n_iter=tune_runs,
                            cv=5,
                            random_state=42,
                            n_jobs=-1,
                            scoring=scoring,
                        )

                    search.fit(bundle.X_train, bundle.y_train)
                    best_model = search.best_estimator_

                    base_pred = bundle.pipeline.predict(bundle.X_test)
                    tuned_pred = best_model.predict(bundle.X_test)

                    base_test = evaluate(problem_type, bundle.y_test, base_pred, "Base Test")
                    tuned_test = evaluate(problem_type, bundle.y_test, tuned_pred, "Tuned Test")

                    compare_df = pd.DataFrame([base_test, tuned_test]).T
                    compare_df.columns = ["Base", "Tuned"]

                    st.success("Tuning complete!")
                    st.write("Best parameters:")
                    st.json(search.best_params_)
                    st.dataframe(compare_df)

                    comp = pd.DataFrame(
                        {
                            "Metric": list(base_test.keys()) + list(tuned_test.keys()),
                            "Value": list(base_test.values()) + list(tuned_test.values()),
                            "Model": ["Base"] * len(base_test) + ["Tuned"] * len(tuned_test),
                        }
                    )
                    fig = px.bar(comp, x="Metric", y="Value", color="Model", barmode="group", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error during hyperparameter tuning: {str(e)}")
                    st.info("Suggestion: Check parameter grid, ensure sufficient data for CV, and try reducing search space or iterations.")

else:
    st.info("Upload a file to begin the AutoML-Flow pipeline.")

# Footer
st.caption("Built with Streamlit + Scikit-learn + Plotly")