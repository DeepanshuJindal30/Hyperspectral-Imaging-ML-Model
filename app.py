
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ---------------------- Load Model ----------------------
@st.cache_resource
def load_trained_model():
    model = load_model('model.h5')
    return model

# ---------------------- Function to Preprocess Data ----------------------
def preprocess_data(df):
    df.columns = df.columns.str.strip().str.lower()
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler

# ---------------------- PCA Function ----------------------
def perform_pca(data):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    return pca_result, explained_variance

# ---------------------- Prediction Function ----------------------
def predict(model, data):
    predictions = model.predict(data)
    return predictions

# ---------------------- Streamlit UI ----------------------
def main():
    st.title("🌽 Hyperspectral Imaging Data Analysis")

    st.sidebar.title("🔍 Options")

    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Raw Data")
        st.dataframe(df.head())

        data, scaler = preprocess_data(df)
        st.write("### Preprocessed Data")
        st.dataframe(pd.DataFrame(data, columns=df.columns))

        pca_result, explained_variance = perform_pca(data)
        st.write(f"### PCA Explained Variance: {explained_variance}")

        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], s=20, alpha=0.7, cmap='viridis')
        ax.set_title('PCA Result')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter)
        st.pyplot(fig)

        model = load_trained_model()

        if st.sidebar.button("Predict DON Concentration"):
            predictions = predict(model, data)
            df['Predicted DON Concentration'] = predictions
            st.write("### Predictions")
            st.dataframe(df[['Predicted DON Concentration']])

            if 'vomitoxin_ppb' in df.columns:
                fig, ax = plt.subplots()
                ax.scatter(df['vomitoxin_ppb'], df['Predicted DON Concentration'], alpha=0.7, edgecolors=(0, 0, 0))
                ax.set_title('Actual vs Predicted DON Concentration')
                ax.set_xlabel('Actual DON Concentration')
                ax.set_ylabel('Predicted DON Concentration')
                ax.plot([df['vomitoxin_ppb'].min(), df['vomitoxin_ppb'].max()],
                        [df['vomitoxin_ppb'].min(), df['vomitoxin_ppb'].max()],
                        color='red')
                st.pyplot(fig)

            st.sidebar.download_button(
                label="📥 Download Predictions",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='predictions.csv',
                mime='text/csv'
            )

if __name__ == "__main__":
    main()
