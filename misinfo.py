import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Streamlit app configuration
st.set_page_config(page_title="Misinformation Detection App", page_icon="📰")
st.title("Misinformation Detection App")
st.write("Enter a tweet to predict if it contains misinformation.")

# Load the saved model and tokenizer
@st.cache_resource
def load_resources():
    model = load_model("bilstm_model_missinfo.h5")
    with open("tokenizer_missinfo.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# Define max_len (must match the value used during training)
max_len = 100

# Input form
with st.form("tweet_form"):
    tweet_input = st.text_area("Enter your tweet here:", height=150)
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        # Preprocess the input tweet
        tweet_seq = tokenizer.texts_to_sequences([tweet_input])
        tweet_pad = pad_sequences(tweet_seq, maxlen=max_len)
        
        # Make prediction
        prediction = model.predict(tweet_pad, verbose=0)
        pred_class = (prediction > 0.5).astype(int)[0][0]
        confidence = prediction[0][0]

        # Display results
        st.subheader("Prediction Result:")
        if pred_class == 1:
            st.error("⚠️ This tweet likely contains misinformation.")
            st.write(f"Confidence: {confidence:.2%}")
        else:
            st.success("✅ This tweet is likely safe.")
            st.write(f"Confidence: {1 - confidence:.2%}")

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
st.write("Built with Streamlit and TensorFlow | Model trained on misinformation dataset")
