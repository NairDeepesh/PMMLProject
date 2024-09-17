from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Retrieve form data and print it
        try:
            time_in_cycles = request.form.get('time_in_cycles')
            T50 = request.form.get('T50')
            P30 = request.form.get('P30')
            Ps30 = request.form.get('Ps30')
            Nc = request.form.get('Nc')
            NRf = request.form.get('NRf')
            phi = request.form.get('phi')
            BPR = request.form.get('BPR')
            W32 = request.form.get('W32')
            htBleed = request.form.get('htBleed')

            print(f"Received data - time_in_cycles: {time_in_cycles}, T50: {T50}, P30: {P30}, Ps30: {Ps30}, Nc: {Nc}, NRf: {NRf}, phi: {phi}, BPR: {BPR}, W32: {W32}, htBleed: {htBleed}")

            # Ensure all fields are filled out
            if None in [time_in_cycles, T50, P30, Ps30, Nc, NRf, phi, BPR, W32, htBleed]:
                raise ValueError("All fields must be filled out")

            data = CustomData(
                time_in_cycles=int(time_in_cycles),
                T50=float(T50),
                P30=float(P30),
                Ps30=float(Ps30),
                Nc=float(Nc),
                NRf=float(NRf),
                phi=float(phi),
                BPR=float(BPR),
                W32=float(W32),
                htBleed=int(htBleed)
            )

            pred_df = data.get_data_as_data_frame()
            print(f"Data prepared for prediction: {pred_df}")

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print(f"Prediction results: {results}")

            return render_template('home.html', results=results[0])

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return render_template('home.html', error=str(ve))

        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', error="Something went wrong. Please check your inputs and try again.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
