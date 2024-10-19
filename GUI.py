import tkinter as tk
from tkinter import messagebox
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib

model = joblib.load('random_forest_model.pkl')

features = ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat',
       'merch_long']

class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title("Fraud Prediction")

        self.labels = {}
        self.entries = {}

        for feature in features:
            label = tk.Label(master, text=feature)
            label.pack()
            entry = tk.Entry(master)
            entry.pack()
            self.labels[feature] = label
            self.entries[feature] = entry

        self.predict_button = tk.Button(master, text="Predict", command=self.predict)
        self.predict_button.pack()

    def predict(self):
        try:
            # Collect user input
            input_data = [float(self.entries[feature].get()) for feature in features]
            input_array = np.array(input_data).reshape(1, -1)

            # Predicting the value
            prediction = model.predict(input_array)
            messagebox.showinfo("Prediction Result", f"Fraud Prediction: {'Fraud' if prediction[0] == 1 else 'Not Fraud'}")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
