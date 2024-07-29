import pickle
from flask import Flask, render_template, request, redirect, url_for

# Load the model and scaler (assuming they're in the 'models' directory)
try:
    with open('models/air_quality_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or scaler not found. Ensure 'air_quality_model.pkl' and 'scaler.pkl' are in the 'models' directory.")
    # Consider adding error handling or redirection to an error page here

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML form

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract form values and convert to floats (handle potential errors)
            T = float(request.form["T"])
            TM = float(request.form["TM"])
            Tm = float(request.form["Tm"])
            SLP = float(request.form["SLP"])
            H = float(request.form["H"])
            VV = float(request.form["VV"])
            V = float(request.form["V"])
            VM = float(request.form["VM"])

            # Prepare the data for prediction (consider data transformation if needed)
            features = [[T, TM, Tm, SLP, H, VV, V, VM]]

            # Scale the data using the same scaler used for training
            features_scaled = scaler.transform(features)

            # Make the prediction
            prediction = model.predict(features_scaled)[0]  # Access the first element

            # Determine air quality description and health advisory based on prediction
            air_quality_description = ""
            health_advisory = ""

            if prediction <= 50:
                air_quality_description = "Good"
                health_advisory = "Air quality is considered satisfactory, and air pollution poses little or no risk."
            elif prediction <= 100:
                air_quality_description = "Moderate"
                health_advisory = "Air quality is acceptable; however, for some pollutants, there may be a moderate health concern for a very small number of people."
            elif prediction <= 150:
                air_quality_description = "Unhealthy for Sensitive Groups"
                health_advisory = "Members of sensitive groups may experience health effects. The general public is not likely to be affected."
            elif prediction <= 200:
                air_quality_description = "Unhealthy"
                health_advisory = "Everyone may begin to experience health effects; members of sensitive groups may experience more serious health effects."
            elif prediction <= 300:
                air_quality_description = "Very Unhealthy"
                health_advisory = "Health alert: everyone may experience more serious health effects."
            else:
                air_quality_description = "Hazardous"
                health_advisory = "Health warnings of emergency conditions. The entire population is more likely to be affected."

            # Prepare additional data for result.html
            additional_data = {
                "air_quality_description": air_quality_description,
                "health_advisory": health_advisory
            }

            return render_template('result.html', prediction=prediction, additional_data=additional_data)
        except (ValueError, KeyError) as e:
            # Handle potential errors during data conversion or prediction
            print(f"Error: {e}")
            return redirect(url_for('error'))  # Redirect to an error page if an error occurs

@app.route('/error')
def error():
    return render_template('error.html')  # Display an error message

if __name__ == '__main__':
    app.run(debug=True)
