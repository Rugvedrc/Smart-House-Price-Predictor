from flask import Flask, render_template, request
from predictor import predict_with_explain

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    explanation = None
    shap_waterfall_plot = None

    if request.method == 'POST':
        try:
            input_data = {
                'number of bedrooms': float(request.form['bedrooms']),
                'number of bathrooms': float(request.form['bathrooms']),
                'lot area': float(request.form['lot_area']),
                'number of floors': float(request.form['floors']),
                'waterfront present': int(request.form['waterfront']),
                'number of views': int(request.form['views']),
                'condition of the house': int(request.form['condition']),
                'grade of the house': int(request.form['grade']),
                'Area of the house(excluding basement)': float(request.form['living_area']),
                'Lattitude': float(request.form['latitude']),
                'Longitude': float(request.form['longitude']),
                'living_area_renov': float(request.form['living_area_renov']),
                'lot_area_renov': float(request.form['lot_area_renov']),
                'Number of schools nearby': int(request.form['schools']),
                'Distance from the airport': float(request.form['distance']),
                'house_age': int(request.form['house_age']),
                'was_renovated': int(request.form['renovated']),
                'total_area': float(request.form['total_area'])
            }

            # Get prediction and explanations
            predicted_price, explanation, shap_waterfall_plot = predict_with_explain(input_data)

        except Exception as e:
            predicted_price = f"An error occurred: {e}"

    return render_template('index.html',
                           predicted_price=predicted_price,
                           explanation=explanation,
                           shap_waterfall_plot=shap_waterfall_plot)

@app.route('/model-metrics')
def model_metrics():
    import pandas as pd
    df = pd.read_csv('model_metrics.csv')
    return render_template('metrics.html', table=df.to_html(classes='table table-bordered table'))


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Render sets this
    app.run(host='0.0.0.0', port=port, debug=True)
