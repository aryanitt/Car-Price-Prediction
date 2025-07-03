# 🚗 Car Price Prediction

A complete end-to-end **Machine Learning web application** that predicts the price of a car based on its specifications such as dimensions, engine details, and categorical features like brand or fuel system. This project demonstrates a full ML lifecycle from data preprocessing and training to deployment via a web interface using **Flask**.

---

##  Key Features

- ⚙️ **Automated pipeline** for data ingestion, preprocessing, model training
- 📊 Uses **OneHotEncoder**, **StandardScaler**, and **PCA** in preprocessing
- 🤖 Supports multiple regression models (e.g., `LinearRegression`, `RandomForestRegressor`)
- 🌐 Interactive web interface using Flask
- 📥 Accepts detailed car specifications as input
- 🎯 Outputs an instant **predicted car price**

---

##  Machine Learning Workflow

- **Data Processing:** 
  - One-hot encoding of categorical features
  - Standard scaling of numerical values
  - Dimensionality reduction using PCA (optional)
  
- **Models Used:**
  - Linear Regression
  - Random Forest
  - Additional models can be plugged in

- **Deployment:** 
  - Flask serves predictions via an HTML form
  - Model and transformer artifacts saved and reused using `joblib` or `pickle`

---

##  Project Structure
Car_Price_Prediction/
│
├── artifacts/ # Stores trained models and transformers
├── notebook/ # Jupyter notebooks for EDA and visualization
├── src/
│ ├── components/ # Modules for ingestion, transformation, training
│ ├── pipeline/ # Prediction and training pipelines
│ ├── logger_config.py # Logging utility
│ ├── exception_config.py # Custom exception handling
│ └── utils.py # Helper functions (e.g., save/load models)
│
├── templates/ # HTML templates (index.html, result.html)
├── static/ # (Optional) CSS, JS, and image files
├── app.py # Entry point for the Flask web app
├── requirements.txt # List of Python dependencies
└── README.md # Project documentation

