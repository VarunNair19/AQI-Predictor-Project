# AQI Predictor: Air Quality Index Prediction System

This project implements a Machine Learning model to predict the Air Quality Index (AQI) for specific locations around Pune. Instead of requiring manual pollutant inputs, it uses historical data trends to forecast AQI and pollutant levels for a given date and place.

## 📂 Project Structure

* **`MAIN 2.py`** (or `main.py`): The core application script. It handles data processing, model training, and launches the Graphical User Interface (GUI).
* **`PNQ_AQI_sorted.csv`**: The dataset containing historical air quality data (Date, Location, SO2, NOx, RSPM, AQI).
* **`aqi_predictor_model.pkl`**: The trained Random Forest model (generated automatically).
* **`AQI.png`**: (Required) Logo image used in the GUI application.

## 🚀 Features

1.  **Smart Data Processing:**
    * Cleans raw data and standardizes location names (e.g., 'MPCB-KR' → 'Karve Road').
    * Fills missing values using forward/backward filling methods.
2.  **Location-Based Prediction:**
    * Users select a specific area (e.g., Swargate, Bhosari, Karve Road) from a dropdown list.
    * The system calculates the *recent 7-day average* of pollutants for that location to use as model inputs.
3.  **Model & Evaluation:**
    * **Algorithm:** Random Forest Regressor.
    * **Metrics:** Calculates R² Score and Mean Absolute Error (MAE) during training.
4.  **Interactive GUI:**
    * Built with Tkinter for easy usage.
    * **Input:** Location & Date (DD-MM-YYYY).
    * **Output:** Predicted AQI + Estimated levels for SO2, NOx, and RSPM.
    * **Visuals:** Generates a bar chart showing the predicted values.

## 🛠️ Installation & Requirements

Ensure you have Python installed. Install the required dependencies:

```bash
pip install pandas matplotlib seaborn scikit-learn pillow joblib
or download requirements.txt
```