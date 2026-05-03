from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model/final_model (1).pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
columns = pickle.load(open("model/columns.pkl", "rb"))

input_data = np.zeros(len(columns))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("ROUTE HIT")

        bhk = int(request.form['bhk'])
        size = int(request.form['size'])
        bathroom = int(request.form['bathroom'])
        floor = int(request.form['floor'])

        city = request.form['city']
        furnishing = request.form['furnishing']

        print("Inputs:", bhk, size, bathroom, floor, city, furnishing)

        input_data = np.zeros(len(columns))

        input_dict = {
            "BHK": bhk,
            "Size": size,
            "Bathroom": bathroom,
            "Floor_Num": floor
        }

        for i, col in enumerate(columns):
            if col in input_dict:
                input_data[i] = input_dict[col]

        city_col = f"City_{city}"
        furn_col = f"Furnishing Status_{furnishing}"

        if city_col in columns:
            input_data[columns.get_loc(city_col)] = 1

        if furn_col in columns:
            input_data[columns.get_loc(furn_col)] = 1

        print("Feature vector ready")

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        print("Scaled input ready")

        pred_log = model.predict(input_scaled)
        print("Raw prediction:", pred_log)

        pred = np.expm1(pred_log)
        print("Final prediction:", pred)

        return f"Prediction: ₹{int(pred[0])}"

    except Exception as e:
        print("ERROR OCCURRED:", e)
        return f"ERROR: {e}"

if __name__ == "__main__":
    app.run(debug=True)