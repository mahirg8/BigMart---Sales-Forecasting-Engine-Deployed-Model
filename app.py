import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.sav')

# Function to process input and predict sales
def predict_sales(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_establishment_year):
    # Mapping the categorical inputs (based on the notebook logic)
    outlet_identifiers = ['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']
    outlet_sizes = {'High': 2, 'Medium': 1, 'Small': 0}
    outlet_types = {'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3, 'Grocery Store': 0}

    # Convert inputs into the format required by the model
    p1 = float(item_mrp)
    p2 = outlet_identifiers.index(outlet_identifier)  # Index in the list
    p3 = outlet_sizes[outlet_size]  # Size mapping
    p4 = outlet_types[outlet_type]  # Type mapping
    p5 = 2024 - int(outlet_establishment_year)  # Age calculation (assuming the current year is 2024)

    # Make prediction using the model
    prediction = model.predict(np.array([[p1, p2, p3, p4, p5]]))
    pred = float(prediction[0])
    
    # Calculate the range
    lower_bound = round(pred - 713.57, 2)
    upper_bound = round(pred + 713.57, 2)
    
    return lower_bound, upper_bound

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        item_mrp = float(request.form['item_mrp'])
        outlet_identifier = request.form['outlet_identifier']
        outlet_size = request.form['outlet_size']
        outlet_type = request.form['outlet_type']
        outlet_establishment_year = request.form['outlet_establishment_year']
        
        # Call the prediction function
        lower_bound, upper_bound = predict_sales(item_mrp, outlet_identifier, outlet_size, outlet_type, outlet_establishment_year)
        
        # Create the prediction message
        message = f'Sales Value is between ${lower_bound} and ${upper_bound}'
        
        return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)
