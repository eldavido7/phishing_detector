from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import requests
import os
import numpy as np
import pandas as pd
import re
import whois
import datetime
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained Keras model
model = load_model("phishing_model.h5")

# Define a scaler (ensure it matches the one used during training)
scaler = MinMaxScaler(feature_range=(0, 1))


# Function to extract features from the URL
def extract_features(url):
    features = []

    # 1. having_IP_Address
    features.append(1 if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", url) else 0)

    # 2. URL_Length (scaled)
    url_length = len(url)
    features.append(url_length / 100 if url_length < 100 else 1)  # Normalize

    # 3. Shortening_Service
    shortening_services = ("bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co")
    features.append(1 if any(service in url for service in shortening_services) else 0)

    # 4. having_At_Symbol
    features.append(1 if "@" in url else 0)

    # 5. double_slash_redirecting
    features.append(1 if "//" in url[7:] else 0)

    # 6. Prefix_Suffix (-1 if '-' in domain name, else 1)
    domain = re.findall(r"://([^/]+)/?", url)
    domain = domain[0] if domain else url
    features.append(1 if "-" in domain else 0)

    # 7. having_Sub_Domain (normalized)
    subdomain_count = domain.count(".")
    features.append(min(subdomain_count / 3, 1))  # Normalize

    # 8. SSLfinal_State (default to 0 if unknown)
    features.append(1 if "https" in url else 0)

    # 9. Domain_registeration_length (Using WHOIS lookup, scaled)
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        days_left = (
            (expiration_date - datetime.datetime.today()).days if expiration_date else 0
        )
        features.append(min(days_left / 365, 1))  # Normalize
    except:
        features.append(0)

    # 10-30: Placeholder features for consistency
    features.extend([0] * 21)  # Ensure total feature count matches training data

    return np.array(features).reshape(1, -1)


DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"  # Replace if needed
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Ensure this is set in env variables


def query_deepseek(url):
    prompt = (
        "You are an AI-powered phishing detection tool trained to identify sophisticated phishing attempts. "
        "Using your knowledge of phishing techniques, phishing URLs, and machine learning approaches like Keras, "
        "analyze the following URL and classify it as either 'Phishing' or 'Legitimate'. "
        "Additionally, provide a confidence score from 1 to 10, where 10 means very high confidence. "
        "Finally, explain your decision in one short sentence. "
        "Format your response strictly like this (no extra text):\n\n"
        "Phishing/Legitimate | Confidence (out of 10) | Short reason\n\n"
        f"URL: {url}\n\n"
        "Answer:"
    )

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "deepseek/deepseek-chat:free",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 100,
    }

    response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)

    if response.status_code != 200:
        raise Exception(f"DeepSeek API call failed: {response.text}")

    message = response.json()["choices"][0]["message"]["content"].strip()

    return message


def parse_deepseek_response(response_text):
    try:
        status, confidence, reason = response_text.split("|")
        status = status.strip()
        confidence = int(confidence.strip())
        reason = reason.strip()

        is_phishing = status.lower() == "phishing"

        return {"confidence": confidence, "isPhishing": is_phishing, "reason": reason}
    except Exception:
        # Fallback in case response is malformed
        return {
            "confidence": 5,
            "isPhishing": True,
            "reason": "Unable to parse response. Try again.",
        }


@app.route("/")
def home():
    return render_template("index.html")  # Serves index.html


@app.route("/check-url", methods=["POST"])
def check_url():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "URL is required"}), 400

    try:
        deepseek_response = query_deepseek(url)
        result = parse_deepseek_response(deepseek_response)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
