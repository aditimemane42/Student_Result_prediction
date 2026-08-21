from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "random_model(4).pkl"
model = joblib.load(MODEL_PATH)


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Random Forest Prediction</title>

    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        body {
            min-height: 100vh;
            background: linear-gradient(135deg, #141e30, #243b55);
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: white;
            border-radius: 20px;
            padding: 35px;
            box-shadow: 0 20px 50px rgba(0,0,0,0.3);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            color: #1d3557;
            font-size: 32px;
            margin-bottom: 10px;
        }

        .header p {
            color: #777;
            font-size: 15px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
        }

        .input-box {
            display: flex;
            flex-direction: column;
        }

        .input-box label {
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }

        .input-box input {
            padding: 13px;
            border: 1px solid #ccc;
            border-radius: 10px;
            font-size: 15px;
            outline: none;
            transition: 0.3s;
        }

        .input-box input:focus {
            border-color: #457b9d;
            box-shadow: 0 0 5px rgba(69,123,157,0.4);
        }

        .button-box {
            margin-top: 28px;
            text-align: center;
        }

        button {
            width: 100%;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #1d3557, #457b9d);
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
            transition: 0.3s;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(29,53,87,0.3);
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            background: #eef6ff;
            border: 1px solid #b9dcff;
        }

        .result h2 {
            color: #1d3557;
            margin-bottom: 8px;
        }

        .prediction {
            font-size: 28px;
            font-weight: bold;
            color: #e63946;
        }

        .error {
            margin-top: 20px;
            padding: 15px;
            background: #ffe5e5;
            color: #c1121f;
            border-radius: 10px;
            text-align: center;
        }

        .footer {
            text-align: center;
            margin-top: 25px;
            color: #888;
            font-size: 13px;
        }

        @media(max-width: 650px) {
            .form-grid {
                grid-template-columns: 1fr;
            }

            .container {
                padding: 25px;
            }

            .header h1 {
                font-size: 26px;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1>🌲 Random Forest Prediction</h1>
        <p>Enter the 8 feature values to get the prediction</p>
    </div>

    <form method="POST">

        <div class="form-grid">

            <div class="input-box">
                <label>Feature 1</label>
                <input type="number" step="any" name="feature1"
                       value="{{ values[0] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 2</label>
                <input type="number" step="any" name="feature2"
                       value="{{ values[1] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 3</label>
                <input type="number" step="any" name="feature3"
                       value="{{ values[2] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 4</label>
                <input type="number" step="any" name="feature4"
                       value="{{ values[3] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 5</label>
                <input type="number" step="any" name="feature5"
                       value="{{ values[4] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 6</label>
                <input type="number" step="any" name="feature6"
                       value="{{ values[5] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 7</label>
                <input type="number" step="any" name="feature7"
                       value="{{ values[6] if values else '' }}" required>
            </div>

            <div class="input-box">
                <label>Feature 8</label>
                <input type="number" step="any" name="feature8"
                       value="{{ values[7] if values else '' }}" required>
            </div>

        </div>

        <div class="button-box">
            <button type="submit">🔮 Predict</button>
        </div>

    </form>

    {% if prediction is not none %}
    <div class="result">
        <h2>Prediction Result</h2>
        <div class="prediction">
            {{ prediction }}
        </div>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">
        <strong>Error:</strong> {{ error }}
    </div>
    {% endif %}

    <div class="footer">
        Random Forest Machine Learning Model
    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    values = []

    if request.method == "POST":

        try:
            values = [
                float(request.form["feature1"]),
                float(request.form["feature2"]),
                float(request.form["feature3"]),
                float(request.form["feature4"]),
                float(request.form["feature5"]),
                float(request.form["feature6"]),
                float(request.form["feature7"]),
                float(request.form["feature8"])
            ]

            # Convert input into NumPy array
            input_data = np.array(values).reshape(1, -1)

            # Prediction
            prediction = model.predict(input_data)[0]

            # Display friendly output
            if prediction == 0:
                prediction = "Class 0"
            elif prediction == 1:
                prediction = "Class 1"
            else:
                prediction = str(prediction)

        except Exception as e:
            error = str(e)

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True
    )
