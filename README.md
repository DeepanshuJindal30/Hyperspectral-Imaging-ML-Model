
# 🌽 Hyperspectral Imaging ML Model

## 📌 Overview
This project processes hyperspectral imaging data to predict DON concentration in corn samples using a neural network model.

---

## 📂 Repository Structure
```
├── app.py                   # Streamlit app code
├── ML_Project.ipynb         # Jupyter Notebook with ML code
├── model.h5                 # Trained Neural Network Model
├── data/                    # (Optional) Sample data files
└── README.md                # Project documentation
```

---

## 🚀 Features
✔️ Hyperspectral data preprocessing  
✔️ Dimensionality reduction using PCA and t-SNE  
✔️ Neural network for regression  
✔️ Streamlit app for real-time prediction  
✔️ Export predictions as CSV  

---

## 📥 Installation
Clone the repository:
```bash
git clone https://github.com/<your-username>/hyperspectral-ml-prediction.git
cd hyperspectral-ml-prediction
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Streamlit App
Start the Streamlit app:
```bash
streamlit run app.py
```

---

## 📊 Data Description
- **Features**: Spectral reflectance values across multiple wavelength bands  
- **Target Variable**: DON concentration (continuous)  

---

## 🧠 Model
- Model Type: Neural Network (3 hidden layers)  
- Activation: ReLU  
- Loss Function: MSE  
- Optimizer: Adam  

---

## 📈 Results
| Metric | Value |
|--------|-------|
| **MAE** | 0.12 |
| **RMSE** | 0.25 |
| **R²** | 0.85 |

---

## 💡 Key Insights
- PCA explained ~70% of variance in the first two components.
- Neural network achieved strong predictive performance.
- t-SNE revealed clustering in data patterns.

---

## 📜 License
This project is licensed under the MIT License.

---

## 🤝 Contributing
Pull requests are welcome! For major changes, open an issue first to discuss what you'd like to change.

---

## ⭐ Acknowledgments
Thanks to **ImagoAI** for providing the data and inspiration for this project.
