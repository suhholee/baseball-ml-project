import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from google.cloud import storage
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Baseball Hit Outcome Predictor",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0066cc;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        padding: 1rem;
        border-radius: 10px;
    }
    .single {
        background-color: #4285F4;
        color: white;
    }
    .extra-base-hit {
        background-color: #34A853;
        color: white;
    }
    .home-run {
        background-color: #EA4335;
        color: white;
    }
    .out {
        background-color: #9AA0A6;
        color: white;
    }
    .card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: black;
    }
    .feature-label {
        font-weight: bold;
        color: #333;
    }
    .info-text {
        color: #666;
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 5px 5px 0 5px;
        border-radius: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 200px; /* Set fixed width for tabs */
        white-space: pre-wrap;
        background-color: #dadce0;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding: 10px 16px;
        font-weight: 600;
        color: #333333;
        border: 1px solid #cccccc;
        border-bottom: none;
        text-align: center; /* Center text horizontally */
        display: flex;
        align-items: center; /* Center text vertically */
        justify-content: center; /* Center text horizontally */
    }
    .stTabs [aria-selected="true"] {
        background-color: #0066cc;
        color: white;
        border: 1px solid #0066cc;
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">⚾ Baseball Hit Outcome Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict hit outcomes based on pitch characteristics and swing mechanics</div>', unsafe_allow_html=True)

# Function to download model from GCS
@st.cache_resource
def download_from_gcs(bucket_name, source_blob_name):
    """Download a file from GCS and return as bytes"""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        content = blob.download_as_bytes()
        return content
    
    except Exception as e:
        st.error(f"Error downloading from GCS: {str(e)}")
        return None

# Function to load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    """Load the model and preprocessor from local or GCS"""
    try:
        # Check if we're running locally or in production
        if os.path.exists('models/final/final_model.pkl'):
            # Local development environment
            model = joblib.load('models/final/final_model.pkl')
            preprocessor = joblib.load('models/final/preprocessor.pkl')
            
            # Load the label mapping
            try:
                label_mapping = np.load('models/result_mapping.npy', allow_pickle=True)
                label_mapping = {int(i): str(label) for i, label in enumerate(label_mapping)}
            except FileNotFoundError:
                label_mapping = {
                    0: 'Home Run',
                    1: 'Extra-Base Hit',
                    2: 'Single',
                    3: 'Out'
                }
            
        else:
            # Production environment (GCS)
            bucket_name = 'baseball-ml-data'
            
            # Download model
            model_content = download_from_gcs(bucket_name, 'models/final_model_latest.pkl')
            model = joblib.load(io.BytesIO(model_content))
            
            # Download preprocessor
            preprocessor_content = download_from_gcs(bucket_name, 'models/preprocessor_latest.pkl')
            preprocessor = joblib.load(io.BytesIO(preprocessor_content))
            
            # Download label mapping
            try:
                mapping_content = download_from_gcs(bucket_name, 'models/result_mapping.npy')
                label_mapping = np.load(io.BytesIO(mapping_content), allow_pickle=True)
                label_mapping = {int(i): str(label) for i, label in enumerate(label_mapping)}
            except:
                label_mapping = {
                    0: 'Home Run',
                    1: 'Extra-Base Hit',
                    2: 'Single',
                    3: 'Out'
                }
        
        return model, preprocessor, label_mapping
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Define the prediction function
def predict_outcome(features_df, model, preprocessor):
    """Make predictions using the loaded model"""
    try:
        # Preprocess the features
        X_preprocessed = preprocessor.transform(features_df)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_preprocessed)[0]
            predicted_class = model.predict(X_preprocessed)[0]
            # Ensure predicted_class is a simple integer
            if isinstance(predicted_class, np.ndarray):
                predicted_class = predicted_class.item()
            elif isinstance(predicted_class, np.integer):
                predicted_class = int(predicted_class)
            return predicted_class, probabilities
        else:
            predicted_class = model.predict(X_preprocessed)[0]
            # Ensure predicted_class is a simple integer
            if isinstance(predicted_class, np.ndarray):
                predicted_class = predicted_class.item()
            elif isinstance(predicted_class, np.integer):
                predicted_class = int(predicted_class)
            return predicted_class, None
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Function to get result label from class index
def get_result_label(class_index, label_mapping):
    """Get the result label from the class index"""
    # Convert numpy types to Python int if needed
    if isinstance(class_index, (np.integer, np.ndarray)):
        class_index = int(class_index)
        
    if label_mapping is not None and class_index in label_mapping:
        return label_mapping[class_index]
    else:
        return f"Unknown Outcome (Class {class_index})"

# Function to create a base64 encoded image for downloading
def get_image_download_link(fig, filename="plot.png", text="Download Plot"):
    """Generate a link to download a plot as an image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">{text}</a>'
    return href

# Function to get CSS class for outcome
def get_outcome_class(outcome):
    """Return the CSS class for the outcome"""
    outcome = outcome.lower().replace(" ", "-")
    return outcome

# Load model and preprocessor
model, preprocessor, label_mapping = load_model_and_preprocessor()

if model is None:
    st.warning("⚠️ Model not loaded. Using mock predictions for demonstration.")

# Create tabs for the app - just 2 tabs as requested
tabs = st.tabs(["Predictor", "Contact Me!"])

with tabs[0]:  # Predictor tab
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)
        
        with st.expander("Pitch Characteristics", expanded=True):
            pitch_velocity = st.slider("Pitch Velocity (MPH)", min_value=70.0, max_value=105.0, value=92.0, step=0.1,
                                    help="The speed of the pitch when it leaves the pitcher's hand")
            
            perceived_velocity = st.slider("Perceived Velocity (MPH)", min_value=70.0, max_value=105.0, value=92.5, step=0.1,
                                        help="The effective velocity as perceived by the batter, accounting for extension")
            
            spin_rate = st.slider("Spin Rate (RPM)", min_value=1000, max_value=3500, value=2200, step=10,
                                help="The rate at which the ball spins after being released")
            
            pitch_type = st.selectbox("Pitch Type", 
                                    options=["4-Seam Fastball", "2-Seam Fastball", "Cutter", "Sinker", 
                                            "Slider", "Curveball", "Changeup", "Sweeper", "Split-Finger"],
                                    index=0)
        
        with st.expander("Pitch Location & Release", expanded=True):
            col_loc1, col_loc2 = st.columns(2)
            
            with col_loc1:
                px = st.slider("Horizontal Location (ft)", min_value=-2.0, max_value=2.0, value=0.0, step=0.01,
                            help="Negative values are inside to right-handed batter")
                
                vertical_release = st.slider("Vertical Release (ft)", min_value=3.0, max_value=7.0, value=5.5, step=0.01,
                                        help="Height of pitch release point")
                
                extension = st.slider("Extension (ft)", min_value=5.0, max_value=8.0, value=6.5, step=0.01,
                                    help="Distance from pitching rubber at release")
            
            with col_loc2:
                pz = st.slider("Vertical Location (ft)", min_value=0.5, max_value=5.0, value=2.5, step=0.01,
                            help="Height of the pitch as it crosses home plate")
                
                horizontal_release = st.slider("Horizontal Release (ft)", min_value=-3.0, max_value=3.0, value=0.0, step=0.01,
                                            help="Lateral position of pitch release point")
                
                arm_angle = st.slider("Arm Angle (degrees)", min_value=0.0, max_value=90.0, value=45.0, step=0.1,
                                    help="Angle of the pitcher's arm at release")
        
        with st.expander("Swing Mechanics", expanded=True):
            bat_speed = st.slider("Bat Speed (MPH)", min_value=60.0, max_value=95.0, value=75.0, step=0.1,
                                help="Speed of the bat at the moment of contact")
            
            swing_length = st.slider("Swing Length (ft)", min_value=4.0, max_value=10.0, value=6.5, step=0.01,
                                    help="Distance the bat travels during the swing")
            
            # Derived metrics (calculated automatically)
            swing_efficiency_ratio = bat_speed / swing_length
            speed_differential = bat_speed - pitch_velocity
            
            st.info(f"Swing Efficiency Ratio: {swing_efficiency_ratio:.2f} (Bat Speed / Swing Length)")
            st.info(f"Speed Differential: {speed_differential:.2f} MPH (Bat Speed - Pitch Velocity)")
    
    with col2:
        st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
        
        # Display pitch location visualization
        st.markdown("### Pitch Location")
        
        fig, ax = plt.subplots(figsize=(4, 5))
        
        # Draw strike zone (approximately 17 inches wide, ~1.5 to ~3.5 feet high)
        strike_zone = plt.Rectangle((-0.83, 1.5), 1.66, 2, fill=False, color='black', lw=2)
        ax.add_patch(strike_zone)
        
        # Plot the pitch location
        ax.scatter([px], [pz], color='red', s=200, zorder=10)
        
        # Set axis limits and labels
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(0, 5)
        ax.set_xlabel('Horizontal Location (ft)')
        ax.set_ylabel('Vertical Location (ft)')
        
        # Add labels for orientation
        ax.text(0, -0.2, 'Home Plate', horizontalalignment='center')
        ax.text(-2.5, 2.5, 'Inside (RHB)', horizontalalignment='left')
        ax.text(2.5, 2.5, 'Inside (LHB)', horizontalalignment='right')
        
        st.pyplot(fig)
        
        # Inside/outside, high/low indicators
        inside_pitch = px < 0
        high_pitch = pz > 3.2
        low_pitch = pz < 2.0
        
        # Pitch category
        fastball_types = ['4-Seam Fastball', '2-Seam Fastball', 'Cutter', 'Sinker', 'Split-Finger']
        breaking_types = ['Slider', 'Sweeper', 'Curveball', 'Knuckle Curve', 'Slurve']
        offspeed_types = ['Changeup', 'Forkball', 'Eephus']
        
        if pitch_type in fastball_types:
            pitch_category = 'Fastball'
        elif pitch_type in breaking_types:
            pitch_category = 'Breaking'
        elif pitch_type in offspeed_types:
            pitch_category = 'Offspeed'
        else:
            pitch_category = 'Other'
        
        # Create a predict button
        predict_button = st.button("Predict Outcome", type="primary", use_container_width=True)
        
        if predict_button:
            # Create a feature DataFrame
            features = {
                'pitch_velocity': pitch_velocity,
                'perceived_velocity': perceived_velocity,
                'spin_rate': spin_rate,
                'vertical_release': vertical_release,
                'horizontal_release': horizontal_release,
                'extension': extension,
                'arm_angle': arm_angle,
                'px': px,
                'pz': pz,
                'pitch_type': pitch_type,
                'bat_speed': bat_speed,
                'swing_length': swing_length,
                'inside_pitch': inside_pitch,
                'high_pitch': high_pitch,
                'low_pitch': low_pitch,
                'pitch_category': pitch_category,
                'swing_efficiency_ratio': swing_efficiency_ratio,
                'speed_differential': speed_differential
            }
            
            features_df = pd.DataFrame([features])
            
            # Make prediction if model is loaded, otherwise use mock prediction
            if model is not None:
                predicted_class, probabilities = predict_outcome(features_df, model, preprocessor)
            else:
                # Mock prediction for demonstration
                if pitch_velocity > 95 and inside_pitch and swing_length > 7:
                    predicted_class = 0  # Home Run
                elif bat_speed > 80 and swing_efficiency_ratio > 12:
                    predicted_class = 1  # Extra-Base Hit
                elif bat_speed > 75 and not high_pitch and not low_pitch:
                    predicted_class = 2  # Single
                else:
                    predicted_class = 3  # Out
                
                # Mock probabilities
                probabilities = np.zeros(4)
                probabilities[predicted_class] = 0.6
                remaining = 0.4
                for i in range(4):
                    if i != predicted_class:
                        probabilities[i] = remaining / 3
            
            if predicted_class is not None:
                # Get the predicted outcome label
                predicted_label = get_result_label(predicted_class, label_mapping)
                
                # Display the prediction
                outcome_class = get_outcome_class(predicted_label)
                st.markdown(f'<div class="prediction-result {outcome_class}">{predicted_label}</div>', unsafe_allow_html=True)
                
                # Display probability distribution if available
                if probabilities is not None and label_mapping is not None:
                    # Create a DataFrame for display
                    prob_data = []
                    for i, prob in enumerate(probabilities):
                        if i in label_mapping:
                            label = label_mapping[i]
                            prob_data.append({'Outcome': label, 'Probability': prob})
                    
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Display as text
                    st.markdown("### Outcome Probabilities")
                    
                    # Create a horizontal bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Define colors for each outcome
                    colors = {
                        'Home Run': '#EA4335',
                        'Extra-Base Hit': '#34A853',
                        'Single': '#4285F4',
                        'Out': '#9AA0A6'
                    }
                    
                    bar_colors = [colors.get(outcome, '#9AA0A6') for outcome in prob_df['Outcome']]
                    
                    bars = ax.barh(prob_df['Outcome'], prob_df['Probability'], color=bar_colors)
                    
                    # Add percentage labels
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                              f'{width:.1%}', va='center', fontweight='bold')
                    
                    ax.set_xlim(0, 1.0)
                    ax.set_xlabel('Probability')
                    ax.set_title('Outcome Probabilities')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Provide download link for the plot
                    st.markdown(get_image_download_link(fig, "outcome_probabilities.png", "Download Probability Chart"), 
                              unsafe_allow_html=True)
            else:
                st.error("Error making prediction. Please try different input values.")
        else:
            # Display initial instructions
            st.markdown("""
            <div class="card">
                <h3>How to Use This Tool</h3>
                <p>Adjust the sliders on the left to set pitch and swing characteristics, then click "Predict Outcome" to see results.</p>
                <p>This tool predicts whether a batted ball will result in:</p>
                <ul>
                    <li><strong>Home Run</strong>: Ball hit over the outfield fence</li>
                    <li><strong>Extra-Base Hit</strong>: Double or triple</li>
                    <li><strong>Single</strong>: Batter reaches first base safely</li>
                    <li><strong>Out</strong>: Batter is put out</li>
                </ul>
                <p>The prediction is based on a machine learning model trained on MLB Statcast data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h3>Key Insights</h3>
                <p>Based on our model analysis:</p>
                <ul>
                    <li>Bat speed is the single most important factor for hit outcomes</li>
                    <li>Swing efficiency (bat speed ÷ swing length) is more important than raw swing length</li>
                    <li>Middle-up pitches have the highest home run probability</li>
                    <li>Breaking balls low in the zone result in more outs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with tabs[1]:  # Project Details tab
    # Contact and links
    st.markdown("### Contact & Resources")
    
    st.markdown("""
    **GitHub Repository**: [github.com/suhholee/ml-baseball-outcome-predictor](https://github.com/suhholee/ml-baseball-outcome-predictor)
    
    **Creator**: Suhho Lee
    
    **LinkedIn**: [linkedin.com/in/suhho-lee](https://www.linkedin.com/in/suhho-lee/)
    
    Feel free to reach out with questions or collaboration opportunities!
    """)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd; color: #666;">
    Baseball Hit Outcome Predictor © 2025 | Built with Streamlit by Suhho Lee
</div>
""", unsafe_allow_html=True)

# Run the app with: streamlit run app.py