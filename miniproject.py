import streamlit as st
import os
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Regression Predictor", layout="wide")
st.title("Regression Model Dataset Upload")

# --- Utility Function: Remove Outliers using IQR ---
def remove_outliers_iqr(df, column):
    unchanged_count = 0
    prev_shape = df.shape[0]

    while unchanged_count < 5:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df_filtered = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
        new_shape = df_filtered.shape[0]

        if new_shape == prev_shape:
            unchanged_count += 1
        else:
            unchanged_count = 0
            prev_shape = new_shape
        df = df_filtered

    return df

# --- Upload Dataset ---
file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

if file is not None:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Show dataset info
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text("### Data Types & Null Info:")
    st.text(s)

    # Show summary statistics
    st.write("### Statistical Description:")
    st.dataframe(df.describe(include='all'))

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values.")

        missing_option = st.radio("How do you want to handle missing values?",
                                  ("Drop rows", "Fill with mean (numerical) / mode (categorical)"))

        if missing_option == "Drop rows":
            df.dropna(inplace=True)
            st.success("Rows with missing values dropped.")
        else:
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if df[col].dtype in [np.float64, np.int64]:
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)
            st.success("Missing values filled.")
    else:
        st.success("No missing values found.")


    # Optional: Drop Columns
    drop_cols = st.multiselect("Select any columns you want to drop", df.columns)
    df.drop(columns=drop_cols, inplace=True)
    
    
    # Encode categorical columns
    df_clean = df.copy()
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # --- Multiple Column Boxplots with Type Annotation ---
    st.subheader("Boxplots for Selected Columns")

    # Prepare list of column names with type labels
    col_type_options = []
    col_type_map = {}  # map from label to actual column name

    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            label = f"{col} (numerical)"
        else:
            label = f"{col} (categorical)"
        col_type_options.append(label)
        col_type_map[label] = col

    # Multiselect input
    selected_labels = st.multiselect("Select columns for boxplots:", col_type_options)

    # Filter selected actual column names
    selected_columns = [col_type_map[label] for label in selected_labels]

    # Plotting
    if selected_columns:
        num_plots = len(selected_columns)
        fig, axes = plt.subplots(nrows=(num_plots + 2) // 3, ncols=3, figsize=(15, 4 * ((num_plots + 2) // 3)))
        axes = axes.flatten()

        for i, col in enumerate(selected_columns):
            sns.boxplot(data=df, y=col, ax=axes[i], color='lightblue')
            axes[i].set_title(f"Boxplot: {col}")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        st.pyplot(fig)
    else:
        st.info("Please select at least one column to display boxplots.")


    # --- Outlier Removal ---
    outlier_columns = st.multiselect(
        "Select columns for outlier removal (IQR method)",
        df_clean.select_dtypes(include=np.number).columns
    )

    # Apply IQR filtering
    for col in outlier_columns:
        df_clean = remove_outliers_iqr(df_clean, col)

    st.write(f"Data shape after outlier removal: {df_clean.shape}")

    # --- Boxplots for those selected columns only ---
    if outlier_columns:
        st.subheader("Boxplots After Outlier Removal (Selected Columns Only)")

        fig, axes = plt.subplots(
            nrows=(len(outlier_columns) + 2) // 3,
            ncols=3,
            figsize=(15, 4 * ((len(outlier_columns) + 2) // 3))
        )
        axes = axes.flatten()

        for i, col in enumerate(outlier_columns):
            sns.boxplot(data=df_clean, y=col, ax=axes[i], color='skyblue')
            axes[i].set_title(f'Boxplot: {col}')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        st.pyplot(fig)
    else:
        st.info("No columns selected for outlier removal.")



    # --- Select Target ---
    target = st.selectbox("Select Target Variable", df_clean.columns)

    # --- Histogram Plots with Conditional Binning (No Target Involved) ---
    st.subheader("Feature-wise Histogram")

    # Multiselect options (excluding target)
    hist_cols_labels = []
    hist_col_map = {}

    for col in df.columns:
        if col != target:
            dtype_label = "numerical" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
            label = f"{col} ({dtype_label})"
            hist_cols_labels.append(label)
            hist_col_map[label] = col

    selected_hist_labels = st.multiselect(
        "Select feature columns to view their frequency distribution:",
        hist_cols_labels
    )
    selected_hist_cols = [hist_col_map[label] for label in selected_hist_labels]

    if selected_hist_cols:
        rows = (len(selected_hist_cols) + 2) // 3
        fig_hist, axes = plt.subplots(nrows=rows, ncols=3, figsize=(16, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(selected_hist_cols):
            ax = axes[i]

            if pd.api.types.is_numeric_dtype(df[col]):
                unique_vals = df[col].nunique()

                if unique_vals > 10:
                    binned_col = pd.cut(df[col], bins=10)
                    counts = binned_col.value_counts().sort_index()
                    counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f"{col} (binned) - Frequency")
                    ax.set_xlabel("Bins")
                    ax.set_ylabel("Count")
                else:
                    counts = df[col].value_counts().sort_index()
                    counts.plot(kind='bar', ax=ax, color='steelblue')
                    ax.set_title(f"{col} - Frequency")
                    ax.set_xlabel(col)
                    ax.set_ylabel("Count")
            else:
                counts = df[col].value_counts().sort_index()
                counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title(f"{col} - Frequency")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")

            ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        st.pyplot(fig_hist)
    else:
        st.info("Select one or more columns to display histograms.")






    # Feature-Target split
    X = df_clean.drop(target, axis=1)
    y = df_clean[target]

    # Train/test split
    test_size = st.number_input(f"Enter train test split size", value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Model
    model = Ridge()
    model.fit(X_train, y_train)

        
    # Predict and Evaluate
    y_pred = model.predict(X_test)

    st.subheader("Correlation Matrix Heatmap")

    try:
        numeric_df = df_clean.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig_corr, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix")
        st.pyplot(fig_corr)
    except Exception as e:
        st.error(f"Could not generate correlation matrix: {e}")


    st.subheader("Model Evaluation")
    st.write(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    

    # --- Actual vs Predicted ---
    st.subheader("Actual vs Predicted with Best Fit Line")
    fig_fit, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color='teal', label="Predicted Points")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', label="Perfect Prediction Line")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    ax.set_title("Actual vs Predicted Charges")
    ax.legend()
    st.pyplot(fig_fit)

    # Store label encoders for inverse transforms (if needed)
    encoders = {}
    input_data = []

    # --- Prediction ---
    st.subheader("Make a Prediction")

    for col in X.columns:
        if df[col].dtype == 'object':
            # Use the same LabelEncoder logic
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le

            options = le.classes_
            selected = st.selectbox(f"Enter value for {col}", options)
            encoded_val = le.transform([selected])[0]
            input_data.append(encoded_val)
        else:
            default_val = float(df[col].mode())
            val = st.number_input(f"Enter value for {col}", value=default_val)
            input_data.append(val)

    if st.button("Predict"):
        prediction = model.predict([input_data])[0]
        st.success(f"Predicted {target}: ${prediction:.2f}")
