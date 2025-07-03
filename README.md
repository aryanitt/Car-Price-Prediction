Car Price Prediction
This project is a Machine Learning web application that predicts the price of a car based on its features such as dimensions, engine specs, and categorical details like fuel system or car brand. The project uses Scikit-learn, Pandas, and Flask for end-to-end development — from data preprocessing to model deployment.

Car_Price_Prediction/
│
├── artifacts/                 # Stores trained models, transformers
├── notebook/                  # EDA and visualization notebooks
├── src/
│   ├── components/            # Data ingestion, transformation, model trainer
│   ├── pipeline/              # Prediction pipeline
│   ├── logger_config.py       # Logging setup
│   ├── exception_config.py    # Custom exceptions
│   └── utils.py               # Utility functions
│
├── templates/                 # HTML Templates (home.html, index.html)
├── static/                    # (Optional) Static files like CSS/images
├── app.py                     # Flask application entry point
├── requirements.txt
└── README.md


Features


Automated data ingestion, preprocessing, and 

Uses OneHotEncoder, StandardScaler, and PCA for pipeline processing.

Trained using regression models (e.g.,LinearRegression, RandomForest).

Integrated with Flask to serve model predictions via web UI.

Accepts user input (car specs) and predicts price instantly.
