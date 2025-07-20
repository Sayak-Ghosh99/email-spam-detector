import streamlit as st
import joblib
import os

# Load model and vectorizer
@st.cache_resource
def load_assets():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, 'notebooks', 'spam_classifier.pkl')
    vectorizer_path = os.path.join(BASE_DIR, 'notebooks', 'vectorizer.pkl')
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_assets()

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            max-width: 800px;
            padding: 2rem;
        }
        .header {
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 1rem;
            margin-bottom: 2rem;
        }
        .result-box {
            border-radius: 5px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .spam-result {
            border-left: 4px solid #ff4b4b;
            background-color: #fff5f5;
        }
        .ham-result {
            border-left: 4px solid #4CAF50;
            background-color: #f5fff5;
        }
        .confidence-badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.85rem;
            background-color: #f0f0f0;
            margin-left: 0.5rem;
        }
        .stButton>button {
            width: 100%;
            padding: 0.75rem;
            font-weight: 500;
        }
        .footer {
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #e0e0e0;
            color: #666;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("Email Spam Classifier")
st.markdown("""
    <p style="color: #555;">
        Classify email messages as spam or legitimate using machine learning.
    </p>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input section
st.subheader("Email Content")
user_input = st.text_area(
    "Paste the email message you want to analyze:",
    height=200,
    placeholder="Enter the email content here...",
    label_visibility="collapsed"
)

# Prediction button
if st.button("Analyze Email", type="primary"):
    if not user_input.strip():
        st.warning("Please enter an email message to analyze.")
    else:
        with st.spinner("Analyzing..."):
            # Vectorize and predict
            X_input = vectorizer.transform([user_input])
            prediction = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][prediction]
            
            # Display results
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-box spam-result">
                        <h3>Spam Detection</h3>
                        <p style="font-size: 1.1rem;">
                            <strong>Result:</strong> This message is classified as <span style="color: #ff4b4b;">SPAM</span>
                            <span class="confidence-badge">{proba:.0%} confidence</span>
                        </p>
                        <p style="color: #666;">
                            Recommendation: Mark as spam or delete this message.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-box ham-result">
                        <h3>Spam Detection</h3>
                        <p style="font-size: 1.1rem;">
                            <strong>Result:</strong> This message is classified as <span style="color: #4CAF50;">LEGITIMATE</span>
                            <span class="confidence-badge">{proba:.0%} confidence</span>
                        </p>
                        <p style="color: #666;">
                            Recommendation: This appears to be a genuine message.
                        </p>
                    </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Email Spam Classifier v1.0 · Made by sayak</p>
    </div>
""", unsafe_allow_html=True)



