import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import joblib
import warnings

warnings.filterwarnings("ignore")

class AQIDataProcessor:
    def __init__(self, filepath):
        self.df = self.load_and_clean_data(filepath)
        self.locations = sorted(self.df['Location'].unique())

    def load_and_clean_data(self, filepath):
        df = pd.read_csv(filepath)

        location_rep = {
            'MPCB-KR': 'Karve Road',
            'MPCB-SWGT': 'Swargate',
            'MPCB-BSRI': 'Bhosari',
            'MPCB-NS': 'Nal Stop',
            'MPCB-PMPR': 'Pimpri',
            'Pimpri Chinchwad': 'Chinchwad'
        }
        df['Location'].replace(location_rep, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.sort_values(by='Date', inplace=True)

        numeric_cols = ['SO2 µg/m3', 'Nox µg/m3', 'RSPM µg/m3', 'AQI']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

        df['AQI Rating'] = pd.cut(df['AQI'],
                                  bins=[0, 50, 100, 150, 200, 300, df['AQI'].max()],
                                  labels=['Good', 'Moderate', 'Moderately Unhealthy', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])

        return df


class AQIPredictor:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.model = None
        self.features = ['SO2 µg/m3', 'Nox µg/m3', 'RSPM µg/m3', 'month', 'day_of_week', 'location_encoded']

    def prepare_features(self):
        df = self.data_processor.df.copy()
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['location_encoded'] = df['Location'].astype('category').cat.codes
        return df

    def train_model(self):
        df = self.prepare_features()
        X = df[self.features]
        y = df['AQI']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        print("Model trained. Train R2: {:.3f}, Test R2: {:.3f}, MAE: {:.2f}".format(
            self.model.score(X_train, y_train), self.model.score(X_test, y_test),
            mean_absolute_error(y_test, self.model.predict(X_test))))
        joblib.dump(self.model, 'aqi_predictor_model.pkl')

    def predict_for_date_location(self, date, location):
        if not self.model:
            self.model = joblib.load('aqi_predictor_model.pkl')

        df = self.data_processor.df
        df['location_encoded'] = df['Location'].astype('category').cat.codes
        location_map = dict(zip(df['Location'], df['location_encoded']))

        if location not in location_map:
            raise ValueError(f"Unknown location: {location}")

        recent_df = df[(df['Location'] == location) & (df['Date'] < date)].tail(7)
        so2 = recent_df['SO2 µg/m3'].mean()
        nox = recent_df['Nox µg/m3'].mean()
        rspm = recent_df['RSPM µg/m3'].mean()
        month = date.month
        day_of_week = date.weekday()
        loc_encoded = location_map[location]

        input_df = pd.DataFrame([{
            'SO2 µg/m3': so2,
            'Nox µg/m3': nox,
            'RSPM µg/m3': rspm,
            'month': month,
            'day_of_week': day_of_week,
            'location_encoded': loc_encoded
        }])

        aqi_pred = round(self.model.predict(input_df)[0], 2)
        return aqi_pred, round(so2, 2), round(nox, 2), round(rspm, 2)


class AQIGUI:
    def __init__(self, root, processor, predictor):
        self.root = root
        self.processor = processor
        self.predictor = predictor

        self.root.title("AQI Prediction App")
        self.root.geometry("600x550")
        self.root.configure(bg="white")

        main_frame = tk.Frame(self.root, bg="white")
        main_frame.pack(pady=10)

        logo = Image.open("AQI.png")
        logo = logo.resize((120, 120))
        self.logo_img = ImageTk.PhotoImage(logo)
        logo_label = tk.Label(main_frame, image=self.logo_img, bg="white")
        logo_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Location Dropdown
        tk.Label(main_frame, text="Select Location:", bg="white", font=("Roboto", 10,"bold")).grid(row=1, column=0, sticky="e", padx=10, pady=5)
        self.location_var = tk.StringVar()
        self.location_menu = ttk.Combobox(main_frame, textvariable=self.location_var, values=self.processor.locations, state="readonly", width=25, font=("Robot", 10))
        self.location_menu.grid(row=1, column=1, padx=10, pady=5)

        # Date Input
        tk.Label(main_frame, text="Enter Date (DD-MM-YYYY):", bg="white", font=("Roboto", 10,"bold")).grid(row=2, column=0, sticky="e", padx=10, pady=5)
        self.date_entry = tk.Entry(main_frame, font=("Robot", 10, "bold"), width=28)
        self.date_entry.grid(row=2, column=1, padx=10, pady=5)

        # Predict Button
        self.predict_button = tk.Button(main_frame, text="Predict AQI", command=self.predict_aqi, bg="#4CAF50", fg="white", font=("Robot", 11, "bold"), width=20)
        self.predict_button.grid(row=3, column=0, columnspan=2, pady=15)

        # Results Display
        self.result_label = tk.Label(main_frame, text="", font=("Robot", 11, "bold"), bg="white", justify="left")
        self.result_label.grid(row=4, column=0, columnspan=2, pady=10)

    def predict_aqi(self):
        date_str = self.date_entry.get()
        location = self.location_var.get()

        try:
            date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            aqi, so2, nox, rspm = self.predictor.predict_for_date_location(date_obj, location)

            self.result_label.config(
                text=f"Predicted AQI for {location} on {date_str}\n\nAQI: {aqi}\nSO2: {so2} µg/m3\nNOx: {nox} µg/m3\nRSPM: {rspm} µg/m3")

            self.show_prediction_plot(date_obj, location, aqi, so2, nox, rspm)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_prediction_plot(self, date, location, aqi_pred, so2_pred, nox_pred, rspm_pred):
        data = {
            'Pollutant': ['AQI', 'SO2 µg/m3', 'Nox µg/m3', 'RSPM µg/m3'],
            'Predicted Value': [aqi_pred, so2_pred, nox_pred, rspm_pred]
        }
        pred_df = pd.DataFrame(data)

        plt.figure(figsize=(6, 4))
        sns.barplot(data=pred_df, x='Pollutant', y='Predicted Value', palette='coolwarm')
        plt.title(f"Predicted AQI & Pollutants on {date.strftime('%d-%b-%Y')} in {location}")
        plt.ylabel("Value")
        plt.xlabel("Pollutants")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    processor = AQIDataProcessor("PNQ_AQI_sorted.csv")
    predictor = AQIPredictor(processor)
    predictor.train_model()

    root = tk.Tk()
    app = AQIGUI(root, processor, predictor)
    root.mainloop()

