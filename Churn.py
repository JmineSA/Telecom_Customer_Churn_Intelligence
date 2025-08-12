# Telecom Customer Churn Intelligence Application
# This application predicts customer churn for a telecom company using a machine learning model.
# Imports and Initial Setup


# Core imports
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import random
import shap
import matplotlib.pyplot as plt
import io
import base64
import json
import time
import xgboost as xgb
from datetime import datetime, timedelta
from functools import lru_cache
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple

# Load model and features
@lru_cache(maxsize=1)
def load_cached_model():
    return joblib.load("Telecom_Customer_Churn_Intelligence/APP/churn_model_top15_features.pkl")

@lru_cache(maxsize=1)
def load_cached_features():
    return joblib.load("Telecom_Customer_Churn_Intelligence/APP/top15_features.pkl")

model = load_cached_model()
feature_list = load_cached_features()

# Define class names
CLASS_NAMES = {
    0: "Stayed",
    1: "Churned",
    2: "Joined"
}

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Initialize encoders
contract_encoder = LabelEncoder().fit(["Month-to-month", "One year", "Two year"])
payment_encoder = LabelEncoder().fit(["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
security_encoder = LabelEncoder().fit(["Yes", "No", "No internet service"])
internet_encoder = LabelEncoder().fit(["DSL", "Fiber optic", "Cable", "Unknown"])

# Professional color scheme with animation colors
COLORS = {
    "primary": "#4f46e5",
    "secondary": "#f0f5ff",
    "accent": "#ec4899",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "fire": "#ff6b35",
    "dark": "#1e293b",
    "light": "#f8fafc",
    "animation_primary": "#6366f1",
    "animation_secondary": "#a5b4fc"
}

# History file
HISTORY_FILE = "prediction_history.json"

# Animation configurations
ANIMATION_DURATION = "0.4s"



#Core Functions
# Preprocess input data for model prediction

def preprocess_input(data: Dict) -> pd.DataFrame:
    processed = {
        'Tenure in Months': data['tenure'],
        'Total Charges': data['total_charges'],
        'Total Revenue': data['total_revenue'],
        'Contract': data['contract'],
        'Total Long Distance Charges': data['total_long_distance'],
        'Monthly Charge': data['monthly_charge'],
        'Number of Referrals': data['number_of_referrals'],
        'Age': data['age'],
        'Avg Monthly GB Download': data['avg_monthly_gb_download'],
        'Payment Method': data['payment_method'],
        'Online Security': data['online_security'],
        'Number of Dependents': data['number_of_dependents'],
        'Internet Type': data['internet_type'],
        'Paperless Billing': data['paperless_billing'],
        'Premium Tech Support': data['premium_tech_support']
    }
    
    df = pd.DataFrame(processed, index=[0])
    
    df['Contract'] = contract_encoder.transform(df['Contract'])
    df['Payment Method'] = payment_encoder.transform(df['Payment Method'])
    df['Online Security'] = security_encoder.transform(df['Online Security'])
    df['Internet Type'] = internet_encoder.transform(df['Internet Type'])
    
    df['Paperless Billing'] = df['Paperless Billing'].map({'Yes': 1, 'No': 0})
    df['Premium Tech Support'] = df['Premium Tech Support'].map({'Yes': 1, 'No': 0})
    
    return df[feature_list]

def validate_inputs(data: Dict) -> Optional[List[str]]:
    errors = []
    if not 18 <= data['age'] <= 100:
        errors.append("Age must be between 18 and 100")
    if data['tenure'] <= 0:
        errors.append("Tenure must be positive")
    if data['monthly_charge'] <= 0:
        errors.append("Monthly charge must be positive")
    if data['total_charges'] <= 0:
        errors.append("Total charges must be positive")
    if data['contract'] == "Two year" and data['tenure'] < 24:
        errors.append("Two-year contract requires minimum 24 months tenure")
    if data['internet_type'] != "No" and data['avg_monthly_gb_download'] <= 0:
        errors.append("Internet users must have positive GB download")
    return errors if errors else None

def get_enhanced_shap_plot(processed_data: pd.DataFrame) -> str:
    try:
        shap_values = explainer.shap_values(processed_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.style.use('seaborn')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = np.arange(len(feature_list))
        ax.barh(y_pos, shap_values[0], align='center', 
               color=COLORS['primary'], alpha=0.7)
        
        for i, v in enumerate(shap_values[0]):
            ax.text(v if v >=0 else v - 0.02, i, 
                   f"{v:.3f}", 
                   color='black' if v >=0 else 'white',
                   va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_list)
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        ax.set_title('Feature Importance for Churn Prediction', pad=20)
        ax.axvline(x=0, color=COLORS['dark'], linestyle='--', alpha=0.3)
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%; height:auto;' class='shap-image' />"
    except Exception as e:
        print(f"Error generating SHAP plot: {str(e)}")
        return "<p>Could not generate feature importance visualization</p>"

def create_fire_animation(risk_level: str) -> str:
    if risk_level == "High":
        return """
        <div class="fire-container">
            <div class="fire">
                <div class="fire-left"></div>
                <div class="fire-main"></div>
                <div class="fire-right"></div>
                <div class="fire-bottom"></div>
            </div>
        </div>
        """
    return ""

def plot_historical_trends(history: List[Dict]) -> str:
    try:
        if not history:
            return "<p>No historical data available</p>"
        
        dates = []
        probs = []
        for record in history[-30:]:
            dates.append(datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S"))
            probs.append(float(record['result']['Probability'].strip('%')) / 100)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, probs, marker='o', 
               color=COLORS['primary'], 
               linestyle='-', 
               linewidth=2,
               markersize=8,
               markerfacecolor=COLORS['light'],
               markeredgecolor=COLORS['primary'])
        
        ax.axhspan(0.7, 1, facecolor=COLORS['danger'], alpha=0.1)
        ax.axhspan(0.4, 0.7, facecolor=COLORS['warning'], alpha=0.1)
        ax.axhspan(0, 0.4, facecolor=COLORS['success'], alpha=0.1)
        
        ax.set_title('Historical Churn Probability Trends', pad=20)
        ax.set_ylabel('Churn Probability')
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(True, linestyle=':', alpha=0.7)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%; height:auto;' />"
    except Exception as e:
        print(f"Error generating trend plot: {str(e)}")
        return "<p>Could not generate trend visualization</p>"

def plot_customer_segments(segment_data: Dict) -> str:
    try:
        segments = list(segment_data.keys())
        counts = list(segment_data.values())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [COLORS['primary'], COLORS['danger'], 
                 COLORS['accent'], COLORS['success']]
        bars = ax.bar(segments, counts, color=colors)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        ax.set_title('Customer Segment Distribution', pad=20)
        ax.set_ylabel('Number of Customers')
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%; height:auto;' />"
    except Exception as e:
        print(f"Error generating segment plot: {str(e)}")
        return "<p>Could not generate segment visualization</p>"

def segment_customer(input_data: Dict) -> str:
    if input_data['total_revenue'] > 3000:
        return "High Value"
    elif input_data['contract'] == "Month-to-month":
        return "At Risk"
    elif input_data['number_of_referrals'] >= 3:
        return "Brand Advocate"
    else:
        return "Standard"

def generate_recommendations(prediction: Dict, input_data: Dict) -> List[str]:
    recs = []
    
    if prediction['Risk'] == "High":
        recs.append("⚠️ Immediate retention action recommended")
        if input_data['contract'] == "Month-to-month":
            recs.append("💡 Offer discounted annual contract")
        if input_data['premium_tech_support'] == "No":
            recs.append("🛠 Offer complimentary tech support trial")
    
    if input_data['number_of_referrals'] == 0:
        recs.append("👥 Consider implementing referral program")
        
    if input_data['avg_monthly_gb_download'] > 100:
        recs.append("📶 Suggest upgraded internet plan")
        
    return recs if recs else ["✅ Customer profile looks healthy - no specific actions recommended"]

def get_segment_distribution() -> Dict:
    return {
        "High Value": 15,
        "At Risk": 25,
        "Brand Advocate": 10,
        "Standard": 50
    }

def save_prediction_to_history(input_data: Dict, prediction_result: Dict) -> None:
    try:
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except FileNotFoundError:
            history = []
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input": input_data,
            "result": prediction_result
        }
        history.append(record)
        
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")

def load_history() -> List[Dict]:
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    return history

def format_history_table(history: List[Dict], risk_filter: str = "All", date_range: str = "All Time") -> str:
    if not history:
        return "<div class='empty-history'>No prediction history found</div>"
    
    # Sort history by timestamp (newest first)
    sorted_history = sorted(history, key=lambda x: x['timestamp'], reverse=True)
    
    # Apply filters
    filtered_history = []
    
    # Date range filter
    now = datetime.now()
    if date_range == "Last 7 Days":
        cutoff = now - timedelta(days=7)
    elif date_range == "Last 30 Days":
        cutoff = now - timedelta(days=30)
    elif date_range == "Last 90 Days":
        cutoff = now - timedelta(days=90)
    else:
        cutoff = None
    
    for record in sorted_history:  # Use the sorted history
        record_date = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S")
        
        # Apply date filter
        if cutoff and record_date < cutoff:
            continue
            
        # Apply risk filter
        if risk_filter == "All":
            filtered_history.append(record)
        elif risk_filter == "High Risk" and record['result']['Risk'] == "High":
            filtered_history.append(record)
        elif risk_filter == "Medium Risk" and record['result']['Risk'] == "Medium":
            filtered_history.append(record)
        elif risk_filter == "Low Risk" and record['result']['Risk'] == "Low":
            filtered_history.append(record)
    
    if not filtered_history:
        return f"<div class='empty-history'>No records match the selected filters</div>"
    
    # Generate table HTML
    table_html = """
    <div class="modern-history-table">
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Risk</th>
                    <th>Segment</th>
                    <th>Contract</th>
                    <th>Tenure</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for record in filtered_history[:50]:  # Show only first 50 records (most recent)
        timestamp = datetime.strptime(record['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M")
        probability = record['result']['Probability']
        prediction = record['result']['Prediction']
        risk = record['result']['Risk']
        
        # Determine risk color
        if risk == "High":
            risk_color = COLORS['danger']
            risk_icon = "🔴"
        elif risk == "Medium":
            risk_color = COLORS['warning']
            risk_icon = "🟠"
        else:
            risk_color = COLORS['success']
            risk_icon = "🟢"
        
        # Get segment
        segment = segment_customer(record['input'])
        
        table_html += f"""
        <tr>
            <td>{timestamp}</td>
            <td>{prediction}</td>
            <td>{probability}</td>
            <td style="color: {risk_color}">{risk_icon} {risk}</td>
            <td class="segment-{segment.lower().replace(' ', '-')}">{segment}</td>
            <td>{record['input']['contract']}</td>
            <td>{record['input']['tenure']} months</td>
        </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    </div>
    """
    
    return table_html

def clear_history():
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump([], f)
        return format_history_table([])
    except Exception as e:
        print(f"Error clearing history: {e}")
        return format_history_table(load_history())

def format_modern_results(result):
    fire_animation = create_fire_animation(result["risk_level"])
    
    if result["risk_level"] == "High":
        icon = "🔴"
        pulse_class = "pulse-danger"
    elif result["risk_level"] == "Medium":
        icon = "🟠"
        pulse_class = "pulse-warning"
    else:
        icon = "🟢"
        pulse_class = "pulse-success"
    
    probability_card = f"""
    <div class="modern-card probability-card {pulse_class}" style="border-left: 4px solid {result['risk_color']};">
        <div class="card-header">
            <span class="card-icon">{icon}</span>
            <h3 class="card-title">Churn Probability</h3>
        </div>
        <div class="card-value" style="color: {result['risk_color']};">{result['probability']}</div>
        <div class="card-footer">Risk: <span style="color: {result['risk_color']};">{result['risk_level']}</span></div>
        {fire_animation}
    </div>
    """
    
    prediction_card = f"""
    <div class="modern-card prediction-card slide-in">
        <div class="card-header">
            <h3 class="card-title">Prediction</h3>
        </div>
        <div class="card-value">{result['prediction']}</div>
        <div class="card-footer">Customer Segment: <span class="segment-{result['customer_segment'].lower().replace(' ', '-')}">{result['customer_segment']}</span></div>
    </div>
    """
    
    recommendations_card = f"""
    <div class="modern-card recommendations-card fade-in">
        <div class="card-header">
            <h3 class="card-title">Recommended Actions</h3>
        </div>
        <div class="card-list">
            {"".join([f'<div class="recommendation-item">{rec}</div>' for rec in result["recommendations"]])}
        </div>
    </div>
    """
    
    return f"""
    <div class="modern-results-container">
        <div class="cards-row">
            {probability_card}
            {prediction_card}
        </div>
        {recommendations_card}
    </div>
    """

def enhanced_predict_churn(*args):
    input_data = {
        'age': args[0],
        'tenure': args[1],
        'monthly_charge': args[2],
        'total_charges': args[3],
        'total_revenue': args[4],
        'total_long_distance': args[5],
        'contract': args[6],
        'internet_type': args[7],
        'payment_method': args[8],
        'online_security': args[9],
        'paperless_billing': args[10],
        'premium_tech_support': args[11],
        'number_of_referrals': args[12],
        'avg_monthly_gb_download': args[13],
        'number_of_dependents': args[14]
    }
    
    if errors := validate_inputs(input_data):
        error_result = {
            "error": "Input validation failed",
            "details": errors,
            "status": "validation_error"
        }
        return (error_result, "Error", "Error", "Error", "", "", "", "", "", "")
    
    try:
        processed_data = preprocess_input(input_data)
        probabilities = model.predict_proba(processed_data)
        
        churn_probability = probabilities[0][1]
        predicted_class = np.argmax(probabilities)
        prediction_name = CLASS_NAMES.get(predicted_class, f"Class {predicted_class}")
        
        if churn_probability >= 0.7:
            risk_level = "High"
            risk_color = COLORS["danger"]
        elif churn_probability >= 0.4:
            risk_level = "Medium"
            risk_color = COLORS["warning"]
        else:
            risk_level = "Low"
            risk_color = COLORS["success"]
        
        shap_html = get_enhanced_shap_plot(processed_data)
        
        recommendations = generate_recommendations({
            "Risk": risk_level,
            "Probability": f"{churn_probability:.2%}"
        }, input_data)
        
        segment = segment_customer(input_data)
        
        segment_dist = get_segment_distribution()
        segment_plot_html = plot_customer_segments(segment_dist)
        
        result = {
            "probability": f"{churn_probability:.2%}",
            "prediction": prediction_name,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "recommendations": recommendations,
            "customer_segment": segment,
            "status": "success"
        }
        
        save_prediction_to_history(input_data, {
            "Probability": result["probability"],
            "Prediction": result["prediction"],
            "Risk": result["risk_level"]
        })
        
        history = load_history()
        trends_plot_html = plot_historical_trends(history)
        
        recommendations_str = "\n".join(recommendations)
        
        modern_output = format_modern_results(result)
        
        return (
            modern_output,
            f"{churn_probability:.2%}",
            prediction_name,
            risk_level,
            risk_color,
            shap_html,
            segment,
            recommendations_str,
            trends_plot_html,
            segment_plot_html
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        error_result = {
            "error": "Prediction failed",
            "details": str(e),
            "status": "prediction_error"
        }
        return (error_result, "Error", "Error", "Error", "", "", "", "", "", "")

def generate_random_inputs() -> List:
    return [
        random.randint(18, 80),
        random.randint(1, 72),
        round(random.uniform(30, 150), 2),
        round(random.uniform(100, 5000), 2),
        round(random.uniform(100, 6000), 2),
        round(random.uniform(0, 200), 2),
        random.choice(["Month-to-month", "One year", "Two year"]),
        random.choice(["DSL", "Fiber optic", "Cable", "Unknown"]),
        random.choice(["Bank transfer", "Credit card", "Electronic check", "Mailed check"]),
        random.choice(["Yes", "No", "No internet service"]),
        random.choice(["Yes", "No"]),
        random.choice(["Yes", "No"]),
        random.randint(0, 10),
        random.randint(0, 200),
        random.randint(0, 5)
    ]

def create_footer():
    return """
    <div class="footer">
        <div class="footer-content">
            <div class="footer-section">
                <h4>Version Information</h4>
                <p>Version: 2.1.0 Business Edition</p>
                <p>Release Date: July 2025</p>
                <p>Model Version: 1.2</p>
                <p>Last Trained: 2023-11-15</p>
            </div>
            <div class="footer-section">
                <h4>Model Performance</h4>
                <p>Accuracy: 0.89</p>
                <p>Precision: 0.85</p>
                <p>Recall: 0.80</p>
                <p>F1 Score: 0.82</p>
            </div>
            <div class="footer-section">
                <h4>Key Features</h4>
                <p>• Real-time predictions</p>
                <p>• SHAP value explanations</p>
                <p>• Customer segmentation</p>
                <p>• Retention recommendations</p>
            </div>
            <div class="footer-section">
                <h4>Developer Information</h4>
                <p>Developed by: Lesiba James Kganyago</p>
                <p>Role: Data Scientist</p>
                <p>GitHub Repository: <a href="#" style="color: var(--primary);">github.com/lesibak</a></p>
            </div>
        </div>
        <div class="footer-bottom">
            <p>© 2025 Telecom Churn Prediction System. All rights reserved.</p>
        </div>
    </div>
    """


#Gradio Interface Setup
# Custom CSS
# Custom CSS with animations and fire effects
custom_css = f"""
:root {{
    --primary: {COLORS["primary"]};
    --secondary: {COLORS["secondary"]};
    --accent: {COLORS["accent"]};
    --success: {COLORS["success"]};
    --warning: {COLORS["warning"]};
    --danger: {COLORS["danger"]};
    --fire: {COLORS["fire"]};
    --dark: {COLORS["dark"]};
    --light: {COLORS["light"]};
    --animation-primary: {COLORS["animation_primary"]};
    --animation-secondary: {COLORS["animation_secondary"]};
    --animation-duration: {ANIMATION_DURATION};
    
    --danger-rgb: 239, 68, 68;
    --warning-rgb: 245, 158, 11;
    --success-rgb: 16, 185, 129;
}}

/* Fire animation */
.fire-container {{
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    height: 60px;
    overflow: hidden;
}}

.fire {{
    position: relative;
    width: 60%;
    height: 100%;
    margin: 0 auto;
}}

.fire-left, .fire-main, .fire-right, .fire-bottom {{
    position: absolute;
    border-radius: 50%;
    background: var(--fire);
    box-shadow: 0 0 10px 5px var(--fire);
    animation: flicker 1.5s infinite alternate;
}}

.fire-left {{
    width: 30px;
    height: 30px;
    bottom: 0;
    left: 0;
    animation-delay: 0.5s;
}}

.fire-main {{
    width: 40px;
    height: 40px;
    bottom: 0;
    left: 50%;
    transform: translateX(-50%);
}}

.fire-right {{
    width: 30px;
    height: 30px;
    bottom: 0;
    right: 0;
    animation-delay: 0.7s;
}}

.fire-bottom {{
    width: 20px;
    height: 20px;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    animation-delay: 0.3s;
}}

@keyframes flicker {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.7; transform: scale(0.95); }}
}}

/* Pulse animations */
.pulse-danger {{
    animation: pulse-danger var(--animation-duration) infinite alternate;
    position: relative;
}}

.pulse-warning {{
    animation: pulse-warning var(--animation-duration) infinite alternate;
}}

.pulse-success {{
    animation: pulse-success var(--animation-duration) infinite alternate;
}}

@keyframes pulse-danger {{
    0% {{ box-shadow: 0 0 0 0 rgba(var(--danger-rgb), 0.4); }}
    100% {{ box-shadow: 0 0 0 10px rgba(var(--danger-rgb), 0); }}
}}

@keyframes pulse-warning {{
    0% {{ box-shadow: 0 0 0 0 rgba(var(--warning-rgb), 0.4); }}
    100% {{ box-shadow: 0 0 0 10px rgba(var(--warning-rgb), 0); }}
}}

@keyframes pulse-success {{
    0% {{ box-shadow: 0 0 0 0 rgba(var(--success-rgb), 0.4); }}
    100% {{ box-shadow: 0 0 0 10px rgba(var(--success-rgb), 0); }}
}}

/* Slide-in animation */
.slide-in {{
    animation: slideIn var(--animation-duration) ease-out;
}}

@keyframes slideIn {{
    from {{ transform: translateX(100%); opacity: 0; }}
    to {{ transform: translateX(0); opacity: 1; }}
}}

/* Fade-in animation */
.fade-in {{
    animation: fadeIn var(--animation-duration) ease-in;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to {{ opacity: 1; }}
}}

/* Button hover animation */
button:hover {{
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15) !important;
}}

/* SHAP image animation */
.shap-image {{
    transition: all 0.3s ease;
}}

.shap-image:hover {{
    transform: scale(1.02);
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}}

/* Header animations */
.title-animation {{
    animation: slideInFromTop var(--animation-duration) ease-out;
}}

.subtitle-animation {{
    animation: fadeIn var(--animation-duration) ease-in 0.2s both;
}}

@keyframes slideInFromTop {{
    from {{ transform: translateY(-50px); opacity: 0; }}
    to {{ transform: translateY(0); opacity: 1; }}
}}

/* Main layout */
.gradio-container {{
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}}

/* Page-like tabs */
.tab {{
    padding: 2rem 5% !important;
    min-height: calc(100vh - 120px);
}}

.tab-nav {{
    background: white !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    padding: 0 5% !important;
    border-bottom: none !important;
}}

.tab-button {{
    padding: 1rem 1.5rem !important;
    margin-right: 0.5rem !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.2s !important;
    font-weight: 500 !important;
}}

.tab-button.selected {{
    border-bottom: 3px solid var(--primary) !important;
    background: transparent !important;
    color: var(--primary) !important;
}}

/* Modern Results Styling */
.modern-results-container {{
    width: 100%;
    margin-top: 1.5rem;
}}

.cards-row {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
}}

@media (max-width: 768px) {{
    .cards-row {{
        flex-direction: column;
    }}
}}

.modern-card {{
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    flex: 1;
    position: relative;
    overflow: hidden;
}}

.probability-card {{
    border-top: 1px solid #e5e7eb;
}}

.prediction-card {{
    border-top: 1px solid #e5e7eb;
}}

.card-header {{
    display: flex;
    align-items: center;
    margin-bottom: 1rem;
}}

.card-icon {{
    font-size: 1.5rem;
    margin-right: 0.75rem;
}}

.card-title {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--dark);
    margin: 0;
}}

.card-value {{
    font-size: 2rem;
    font-weight: 700;
    margin: 0.5rem 0;
}}

.probability-card .card-value {{
    font-size: 2.5rem;
}}

.card-footer {{
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
}}

.recommendations-card {{
    border-top: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}}

.card-list {{
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}}

.recommendation-item {{
    padding: 0.75rem;
    background: var(--secondary);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    transition: all 0.2s;
}}

.recommendation-item:hover {{
    transform: translateX(5px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}}

.recommendation-item::before {{
    content: "•";
    color: var(--accent);
    font-weight: bold;
    font-size: 1.2rem;
}}

.segment-high-value {{
    color: var(--success);
    font-weight: 600;
}}

.segment-at-risk {{
    color: var(--danger);
    font-weight: 600;
}}

.segment-brand-advocate {{
    color: var(--accent);
    font-weight: 600;
}}

.segment-standard {{
    color: var(--primary);
    font-weight: 600;
}}

/* Footer styles */
.footer {{
    background: var(--dark);
    color: white;
    padding: 2rem 5%;
    margin-top: 2rem;
}}

.footer-content {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}}

.footer-section h4 {{
    color: white;
    margin-bottom: 1rem;
    font-size: 1rem;
}}

.footer-section p {{
    margin: 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.9rem;
}}

.footer-bottom {{
    text-align: center;
    padding-top: 2rem;
    margin-top: 2rem;
    border-top: 1px solid #334155;
    color: #94a3b8;
    font-size: 0.8rem;
}}

/* Input panel styling */
.input-panel {{
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    border: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
    transition: all 0.3s;
}}

.input-panel:hover {{
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}}

/* History table styles */
.modern-history-table {{
    width: 100%;
    overflow-x: auto;
    margin-top: 1rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

.modern-history-table table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}}

.modern-history-table th {{
    background: var(--primary);
    color: white;
    padding: 0.75rem 1rem;
    text-align: left;
    position: sticky;
    top: 0;
}}

.modern-history-table td {{
    padding: 0.75rem 1rem;
    border-bottom: 1px solid #e5e7eb;
}}

.modern-history-table tr:nth-child(even) {{
    background-color: #f9fafb;
}}

.modern-history-table tr:hover {{
    background-color: var(--secondary);
}}

.empty-history {{
    padding: 2rem;
    text-align: center;
    color: #64748b;
    background: #f8fafc;
    border-radius: 8px;
    margin-top: 1rem;
}}

.danger-button {{
    background: var(--danger) !important;
    color: white !important;
}}

.danger-button:hover {{
    background: var(--danger) !important;
    opacity: 0.9 !important;
}}

/* Buttons */
button {{
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
    margin-right: 0.75rem !important;
    margin-bottom: 0.75rem !important;
}}

button:hover {{
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
}}

.secondary-button {{
    background: var(--secondary) !important;
    color: var(--primary) !important;
    border: 1px solid var(--primary) !important;
}}

/* Responsive adjustments */
@media (max-width: 768px) {{
    .footer-content {{
        grid-template-columns: 1fr;
    }}
    
    .tab {{
        padding: 1.5rem !important;
    }}
}}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Default(primary_hue="indigo")) as app:
    # Header with animation
    with gr.Row(elem_classes="header"):
        gr.Markdown("""
        <div class="header-content">
            <h1 class="title-animation">📊 Telecom Customer Churn Prediction</h1>
            <p class="subtitle-animation">AI-powered analytics to predict customer churn risk and understand key factors</p>
        </div>
        """)
    
    # Customer Profile Tab
    with gr.Tab("🧑 Customer Profile", elem_id="profile-tab"):
        with gr.Column():
            with gr.Row():
                with gr.Column(scale=2, elem_classes="input-panel"):
                    gr.Markdown("### Personal Information")
                    with gr.Row():
                        age = gr.Slider(18, 100, label="Customer Age", value=45, step=1, interactive=True)
                        number_of_dependents = gr.Slider(0, 5, label="Number of Dependents", value=0, step=1, interactive=True)
                    
                    gr.Markdown("### Financial Information")
                    with gr.Row():
                        monthly_charge = gr.Number(label="Monthly Charge ($)", value=70.00, precision=2)
                        total_charges = gr.Number(label="Total Charges ($)", value=840.00, precision=2)
                    with gr.Row():    
                        total_revenue = gr.Number(label="Total Revenue ($)", value=1000.00, precision=2)
                        total_long_distance = gr.Number(label="Long Distance Charges ($)", value=50.00, precision=2)
                
                with gr.Column(scale=1, elem_classes="input-panel"):
                    gr.Markdown("### Service Details")
                    tenure = gr.Slider(1, 72, label="Tenure (months)", value=12, step=1, interactive=True)
                    contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], 
                                         label="Contract Type", value="Month-to-month")
                    internet_type = gr.Dropdown(["DSL", "Fiber optic", "Cable", "Unknown"], 
                                              label="Internet Service", value="Fiber optic")
                    
                    gr.Markdown("### Service Features")
                    payment_method = gr.Dropdown(
                        ["Bank transfer", "Credit card", "Electronic check", "Mailed check"], 
                        label="Payment Method", 
                        value="Credit card"
                    )
                    online_security = gr.Radio(
                        ["Yes", "No", "No internet service"], 
                        label="Online Security", 
                        value="No",
                        interactive=True
                    )
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")
    
    # Usage & Preferences Tab
    with gr.Tab("📊 Usage & Preferences", elem_id="usage-tab"):
        with gr.Column():
            with gr.Row(elem_classes="input-panel"):
                with gr.Column():
                    gr.Markdown("### Usage Metrics")
                    number_of_referrals = gr.Slider(0, 10, label="Number of Referrals", value=0, step=1, interactive=True)
                    avg_monthly_gb_download = gr.Slider(0, 200, label="Monthly GB Download", value=50, step=1, interactive=True)
                
                with gr.Column():
                    gr.Markdown("### Billing & Support")
                    paperless_billing = gr.Radio(
                        ["Yes", "No"], 
                        label="Paperless Billing", 
                        value="Yes",
                        interactive=True
                    )
                    premium_tech_support = gr.Radio(
                        ["Yes", "No"], 
                        label="Premium Tech Support", 
                        value="No",
                        interactive=True
                    )
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn_usage = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn_usage = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn_usage = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")

    # Results Tab
    with gr.Tab("📈 Results & Analysis", elem_id="results-tab"):
        with gr.Column():
            with gr.Row(elem_classes="output-panel"):
                with gr.Column():
                    gr.Markdown("## Prediction Results")
                    output = gr.HTML(label="Analysis Results")
                    
                    with gr.Row():
                        with gr.Column():
                            probability_display = gr.Label(label="Churn Probability", elem_classes="probability-display")
                        with gr.Column():
                            prediction_display = gr.Label(label="Prediction", elem_classes="prediction-display")
                        with gr.Column():
                            risk_display = gr.Label(label="Risk Level")
                    
                    color_store = gr.Textbox(visible=False)
                    
                    gr.Markdown("""
                    ### Risk Interpretation Guide:
                    - <span class='risk-indicator risk-high'>🔴 High Risk</span>: >70% probability of churn
                    - <span class='risk-indicator risk-medium'>🟠 Medium Risk</span>: 40-70% probability
                    - <span class='risk-indicator risk-low'>🟢 Low Risk</span>: <40% probability
                    """)
                    
                    shap_plot = gr.HTML(label="Feature Importance (SHAP)", elem_classes="shap-plot")
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn_results = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn_results = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn_results = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")

    # Customer Analytics Tab
    with gr.Tab("📊 Customer Analytics", elem_id="analytics-tab"):
        with gr.Column():
            with gr.Row():
                with gr.Column(elem_classes="input-panel"):
                    gr.Markdown("### Customer Segmentation")
                    segment_display = gr.Label(label="Customer Segment")
                    
                    gr.Markdown("### Retention Recommendations")
                    recommendations_display = gr.Textbox(
                        label="Suggested Actions",
                        interactive=False,
                        lines=5,
                        elem_classes="recommendation-box"
                    )
                
                with gr.Column(elem_classes="input-panel"):
                    gr.Markdown("### Historical Trends")
                    trends_plot = gr.HTML(label="Churn Probability Over Time")
                    
                    gr.Markdown("### Segment Distribution")
                    segment_plot = gr.HTML(label="Customer Segments")
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn_analytics = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn_analytics = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn_analytics = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")

    # History Tab
    with gr.Tab("📚 Prediction History", elem_id="history-tab"):
        with gr.Column():
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")
                clear_history_btn = gr.Button("🗑️ Clear History", variant="stop", elem_classes="danger-button")
            
            with gr.Row():
                history_filter = gr.Dropdown(
                    ["All", "High Risk", "Medium Risk", "Low Risk"], 
                    label="Filter by Risk Level",
                    value="All"
                )
                date_range = gr.Dropdown(
                    ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"],
                    label="Date Range",
                    value="All Time"
                )
            
            history_output = gr.HTML(label="Prediction History")
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn_history = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn_history = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn_history = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")
    
    # About Tab
    with gr.Tab("ℹ️ About", elem_id="about-tab"):
        with gr.Column():
            with gr.Row():
                with gr.Column(elem_classes="input-panel"):
                    gr.Markdown("""
                    ## Model Information
                    
                    **Version:** 2.1.0 Business Edition  
                    **Release Date:** July 2025  
                    **Model Version:** 1.2  
                    **Last Trained:** 2023-11-15
                    
                    ## Model Performance Metrics
                    
                    - **Accuracy:** 0.89  
                    - **Precision:** 0.85  
                    - **Recall:** 0.80  
                    - **F1 Score:** 0.82  
                    
                    _Note: Confidence intervals are approximated by prediction probability ranges._
                    """)
                
                with gr.Column(elem_classes="input-panel"):
                    gr.Markdown("""
                    ## Key Features
                    
                    - Real-time churn probability predictions
                    - SHAP value explanations for feature importance
                    - Customer segmentation analytics
                    - Actionable retention recommendations
                    - Historical prediction tracking
                    - Responsive dashboard interface
                    
                    ## Developer Information
                    
                    **Developed by:** Lesiba James Kganyago  
                    **Role:** Data Scientist  
                    **GitHub Repository:** [github.com/lesibak](#)
                    """)
            
            # Prediction controls at bottom of tab
            with gr.Row():
                submit_btn_about = gr.Button("🔮 Predict Churn Risk", variant="primary")
                random_btn_about = gr.Button("🎲 Random Inputs", variant="secondary", elem_classes="secondary-button")
                clear_btn_about = gr.Button("🧹 Clear Inputs", variant="secondary", elem_classes="secondary-button")

    # Footer
    gr.HTML(create_footer())

    # Event handlers for all submit buttons
    submit_btn.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )
    
    submit_btn_usage.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )
    
    submit_btn_results.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )
    
    submit_btn_analytics.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )
    
    submit_btn_history.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )
    
    submit_btn_about.click(
        fn=enhanced_predict_churn,
        inputs=[age, tenure, monthly_charge, total_charges, total_revenue,
               total_long_distance, contract, internet_type, payment_method,
               online_security, paperless_billing, premium_tech_support,
               number_of_referrals, avg_monthly_gb_download, number_of_dependents],
        outputs=[output, probability_display, prediction_display, risk_display, 
                color_store, shap_plot, segment_display, recommendations_display,
                trends_plot, segment_plot]
    )

    # Event handlers for random buttons
    random_btn.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    random_btn_usage.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    random_btn_results.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    random_btn_analytics.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    random_btn_history.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    random_btn_about.click(
        fn=generate_random_inputs,
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )

    # Event handlers for clear buttons
    clear_btn.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    clear_btn_usage.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    clear_btn_results.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    clear_btn_analytics.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    clear_btn_history.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )
    
    clear_btn_about.click(
        fn=lambda: [45, 12, 70.00, 840.00, 1000.00, 50.00, 
                   "Month-to-month", "Fiber optic", "Credit card", 
                   "No", "Yes", "No", 0, 50, 0],
        outputs=[age, tenure, monthly_charge, total_charges, total_revenue,
                total_long_distance, contract, internet_type, payment_method,
                online_security, paperless_billing, premium_tech_support,
                number_of_referrals, avg_monthly_gb_download, number_of_dependents]
    )

    # History tab event handlers
    refresh_btn.click(
        fn=lambda risk, date: format_history_table(load_history(), risk, date),
        inputs=[history_filter, date_range],
        outputs=history_output
    )

    clear_history_btn.click(
        fn=clear_history,
        inputs=None,
        outputs=history_output
    )

# Launch the app
if __name__ == "__main__":
    app.launch(share=True, server_port=7860)