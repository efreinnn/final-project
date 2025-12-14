# About final-project

# Turbo.az Car Price Prediction (BMW & Mercedes) 🇦🇿

## 📌 Overview
This project is a machine learning application that predicts vehicle prices based on data from **Turbo.az**, the largest car listing site in Azerbaijan.

The project takes a specialized approach by creating **separate modeling pipelines** for BMW and Mercedes-Benz. This allows for higher accuracy by capturing the unique depreciation curves and feature values of each specific brand.

## ⚙️ Workflow
The project follows a structured data science lifecycle using Jupyter Notebooks:

1.  **Data Scraping:** Extracted raw data from Turbo.az using **BeautifulSoup**.
2.  **Split Processing:**
    * **BMW Pipeline:** Dedicated cleaning, feature engineering, and model training specifically for BMW listings.
    * **Mercedes Pipeline:** Independent cleaning and modeling process optimized for Mercedes listings.
3.  **Modeling:** Tested multiple algorithms (XGBoost, Random Forest, etc.) to find the best fit for each brand.
4.  **Deployment:** Visualized the results and created a prediction interface using **Streamlit**.

## 🛠️ Technologies Used
* **Environment:** Jupyter Notebooks, VS Code
* **Web Scraping:** BeautifulSoup4
* **Data Science:** Pandas, NumPy, Scikit-Learn, XGBoost
* **Web App:** Streamlit

## 📂 Project Structure
* `TURBO AZ.ipynb` - Notebook used to scrape Turbo.az and save data to Excel.
* `BMW Dealer Data.ipynb` - Data cleaning and model training specific to BMW.
* `Mercedes Dealer Data.ipynb` - Data cleaning and model training specific to Mercedes.
* `model.py` - The Streamlit application file for the user interface.
