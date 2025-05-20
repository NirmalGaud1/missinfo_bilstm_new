import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import google.generativeai as genai

# Streamlit app configuration
st.set_page_config(page_title="Fluoride Misinformation Detection App", page_icon="🦷")
st.title("Fluoride Misinformation Detection App")
st.write("Enter a tweet about fluoride or water fluoridation to predict if it contains misinformation.")

# Configure Gemini API
API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load the saved model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model("bilstm_model_missinfo.h5")
    with open("tokenizer_missinfo.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# Define max_len (must match the value used during model training)
max_len = 280  # Adjusted to match Twitter's 280-character limit

# Function to check if tweet is related to fluoride or water fluoridation
def is_fluoride_related(tweet):
    fluoride_keywords = [
        "fluoride", "fluoridation", "water fluoridation", "fluorinated", 
        "fluoride in water", "fluoride treatment", "fluoride toothpaste"
    ]
    tweet_lower = tweet.lower()
    return any(keyword in tweet_lower for keyword in fluoride_keywords)

# Function to validate prediction with Gemini
def validate_with_gemini(tweet, pred_class):
    try:
        prompt = f"Is this statement about fluoride or water fluoridation misinformation? '{tweet}' Provide a brief explanation."
        response = gemini_model.generate_content(prompt)
        # Check if Gemini considers it misinformation
        gemini_result = "misinformation" if "misinformation" in response.text.lower() else "not misinformation"
        # Gemini agrees if its result matches the model's prediction
        model_result = "misinformation" if pred_class == 0 else "not misinformation"
        explanation = response.text
        return gemini_result == model_result, explanation
    except Exception as e:
        return False, f"Error with Gemini API: {str(e)}"

# Input form with 280-character limit
with st.form("tweet_form"):
    tweet_input = st.text_area(
        "Enter your tweet about fluoride or water fluoridation here:",
        height=150,
        max_chars=280  # Twitter's character limit
    )
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    elif not is_fluoride_related(tweet_input):
        st.error("Invalid input: Tweet must be related to fluoride or water fluoridation.")
    else:
        # Preprocess the input tweet
        tweet_seq = tokenizer.texts_to_sequences([tweet_input])
        tweet_pad = pad_sequences(tweet_seq, maxlen=max_len)
        
        # Make prediction with BiLSTM model
        prediction = model.predict(tweet_pad, verbose=0)
        pred_class = (prediction > 0.5).astype(int)[0][0]
        confidence = prediction[0][0]

        # Validate with Gemini
        is_valid, gemini_explanation = validate_with_gemini(tweet_input, pred_class)

        # Display results
        st.subheader("Prediction Result:")
        if is_valid:
            if pred_class == 0:
                st.error("⚠️ This tweet likely contains misinformation about fluoride or water fluoridation.")
                st.write(f"Confidence: {confidence:.2%}")
                st.write(f"Reason:{gemini_explanation}")
            else:
                st.success("✅ This tweet is likely safe regarding fluoride or water fluoridation.")
                st.write(f"Confidence: {(1 - confidence):.2%}")
                st.write(f"Reason:{gemini_explanation}")
        else:
            st.write(f"Model Prediction: {'Misinformation' if pred_class == 0 else 'Safe'} (Confidence: {confidence:.2%})")
            st.write(f"Response: {gemini_explanation}")

# Add some styling and footer
st.markdown("""
    <style>
    .stTextArea textarea {
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("---")
