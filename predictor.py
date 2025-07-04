import joblib
import pandas as pd
import shap
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Column names
feature_columns = [
    'number of bedrooms', 'number of bathrooms', 'lot area', 'number of floors',
    'waterfront present', 'number of views', 'condition of the house', 'grade of the house',
    'Area of the house(excluding basement)', 'Lattitude', 'Longitude',
    'living_area_renov', 'lot_area_renov', 'Number of schools nearby',
    'Distance from the airport', 'house_age', 'was_renovated', 'total_area'
]

# Create SHAP explainer once
explainer = shap.Explainer(model)

def get_feature_friendly_name(feature_name):
    """Convert technical feature names to user-friendly names"""
    name_mapping = {
        'number of bedrooms': 'Bedrooms',
        'number of bathrooms': 'Bathrooms', 
        'lot area': 'Lot Size',
        'number of floors': 'Number of Floors',
        'waterfront present': 'Waterfront Property',
        'number of views': 'Number of Views',
        'condition of the house': 'House Condition',
        'grade of the house': 'Construction Quality',
        'Area of the house(excluding basement)': 'Living Area',
        'Lattitude': 'Location (Latitude)',
        'Longitude': 'Location (Longitude)',
        'living_area_renov': 'Renovated Living Area',
        'lot_area_renov': 'Renovated Lot Area',
        'Number of schools nearby': 'Nearby Schools',
        'Distance from the airport': 'Airport Distance',
        'house_age': 'House Age',
        'was_renovated': 'Recently Renovated',
        'total_area': 'Total Area'
    }
    return name_mapping.get(feature_name, feature_name)

def get_impact_explanation(feature_name, impact_value, feature_value):
    """Generate human-readable explanations for each feature impact"""
    
    explanations = {
        'number of bedrooms': {
            'positive': f"Having {int(feature_value)} bedrooms increases the price - more bedrooms typically mean higher value",
            'negative': f"Having {int(feature_value)} bedrooms decreases the price - this might be too few or too many for the area"
        },
        'number of bathrooms': {
            'positive': f"Having {feature_value} bathrooms increases the price - more bathrooms add convenience and value",
            'negative': f"Having {feature_value} bathrooms decreases the price - this might be insufficient for the house size"
        },
        'lot area': {
            'positive': f"The large lot size ({int(feature_value):,} sqft) significantly increases the price - bigger lots are more valuable",
            'negative': f"The lot size ({int(feature_value):,} sqft) decreases the price - might be too small for the area"
        },
        'house_age': {
            'positive': f"The house age ({int(feature_value)} years) increases the price - it's in the sweet spot of being mature but not too old",
            'negative': f"The house age ({int(feature_value)} years) decreases the price - older houses typically cost less"
        },
        'condition of the house': {
            'positive': f"The good condition (rating: {int(feature_value)}/5) increases the price - well-maintained houses are worth more",
            'negative': f"The house condition (rating: {int(feature_value)}/5) decreases the price - poor condition reduces value"
        },
        'grade of the house': {
            'positive': f"The high construction quality (grade: {int(feature_value)}/13) increases the price - better built homes are more valuable",
            'negative': f"The construction quality (grade: {int(feature_value)}/13) decreases the price - lower quality construction reduces value"
        },
        'Area of the house(excluding basement)': {
            'positive': f"The spacious living area ({int(feature_value):,} sqft) increases the price - larger homes cost more",
            'negative': f"The living area ({int(feature_value):,} sqft) decreases the price - smaller spaces are typically less expensive"
        },
        'waterfront present': {
            'positive': "Being a waterfront property significantly increases the price - waterfront homes are premium properties",
            'negative': "Not being waterfront doesn't add the premium that waterfront properties command"
        },
        'number of views': {
            'positive': f"Having {int(feature_value)} view(s) increases the price - properties with good views are more desirable",
            'negative': f"Having {int(feature_value)} view(s) decreases the price - limited views reduce desirability"
        },
        'Number of schools nearby': {
            'positive': f"Having {int(feature_value)} school(s) nearby increases the price - good for families with children",
            'negative': f"Having {int(feature_value)} school(s) nearby decreases the price - fewer educational options nearby"
        },
        'Distance from the airport': {
            'positive': f"Being {feature_value:.1f} km from the airport increases the price - convenient but not too close to noise",
            'negative': f"Being {feature_value:.1f} km from the airport decreases the price - either too far or too close with noise issues"
        },
        'was_renovated': {
            'positive': "Recent renovations increase the price - updated homes are more valuable and move-in ready",
            'negative': "No recent renovations decreases the price - buyers prefer updated properties"
        },
        'total_area': {
            'positive': f"The total area ({int(feature_value):,} sqft) increases the price - larger total space is more valuable",
            'negative': f"The total area ({int(feature_value):,} sqft) decreases the price - smaller total space is less valuable"
        }
    }
    
    # Default explanation if feature not found
    if feature_name not in explanations:
        direction = "increases" if impact_value > 0 else "decreases"
        return f"This feature {direction} the price based on its value of {feature_value}"
    
    return explanations[feature_name]['positive' if impact_value > 0 else 'negative']

def predict_with_explain(input_dict):
    """
    Predicts house price and generates user-friendly explanations.
    """
    input_df = pd.DataFrame([input_dict], columns=feature_columns)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    shap_values = explainer(scaled_input)

    # Get base value (average prediction)
    base_value = explainer.expected_value
    
    # Create detailed explanations
    explanations = []
    for i, feature in enumerate(feature_columns):
        impact = shap_values.values[0][i]
        if abs(impact) > 1000:  # Only show significant impacts
            explanations.append({
                'feature': feature,
                'friendly_name': get_feature_friendly_name(feature),
                'impact': impact,
                'feature_value': input_dict[feature],
                'explanation': get_impact_explanation(feature, impact, input_dict[feature]),
                'impact_amount': abs(impact)
            })
    
    # Sort by impact magnitude
    explanations.sort(key=lambda x: x['impact_amount'], reverse=True)
    
    # Take top 6 most impactful features
    top_explanations = explanations[:6]
    
    # Create summary
    total_positive = sum(exp['impact'] for exp in top_explanations if exp['impact'] > 0)
    total_negative = sum(exp['impact'] for exp in top_explanations if exp['impact'] < 0)
    
    summary = {
        'base_price': base_value,
        'total_positive_impact': total_positive,
        'total_negative_impact': abs(total_negative),
        'final_prediction': prediction,
        'explanations': top_explanations
    }

    # Create simple waterfall plot
    waterfall_base64 = create_simple_waterfall_plot(summary)

    return round(prediction, 2), summary, waterfall_base64

def create_simple_waterfall_plot(summary):
    """Create a simple, clean waterfall plot showing price components"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Data for the plot
    categories = ['Base Price']
    values = [summary['base_price']]
    colors = ['#2E86AB']
    
    # Add positive impacts
    for exp in summary['explanations']:
        if exp['impact'] > 0:
            categories.append(f"+ {exp['friendly_name']}")
            values.append(exp['impact'])
            colors.append('#A23B72')
    
    # Add negative impacts  
    for exp in summary['explanations']:
        if exp['impact'] < 0:
            categories.append(f"- {exp['friendly_name']}")
            values.append(exp['impact'])
            colors.append('#F18F01')
    
    # Add final price
    categories.append('Final Price')
    values.append(summary['final_prediction'])
    colors.append('#C73E1D')
    
    # Create the plot
    bars = ax.bar(categories, values, color=colors, alpha=0.8)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (height*0.01),
                f'₹{value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the plot
    ax.set_title('How We Calculated Your House Price', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Price Impact (₹)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Format y-axis to show values in lakhs/crores
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L' if x >= 100000 else f'₹{x/1000:.0f}K'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save plot to base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    
    waterfall_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return waterfall_base64