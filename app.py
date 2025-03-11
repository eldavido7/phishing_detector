from flask import Flask, render_template, request, jsonify
import pandas as pd
from tensorflow import keras
from urllib.parse import urlparse
import tldextract
import requests
import os

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model("phishing_model.keras")

# DeepSeek API Configuration
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # Ensure this is set in env variables


# Feature extraction function
def extract_features(url):
    parsed_url = urlparse(url)
    tld_info = tldextract.extract(url)

    features = {
        "URL": url,
        "URLLength": len(url),
        "Domain": parsed_url.netloc,
        "DomainLength": len(parsed_url.netloc),
        "IsDomainIP": parsed_url.netloc.replace(".", "").isdigit(),
        "TLD": tld_info.suffix,
        "URLSimilarityIndex": 0,  # Placeholder
        "CharContinuationRate": 0,  # Placeholder
        "TLDLegitimateProb": 0,  # Placeholder
        "URLCharProb": 0,  # Placeholder
        "TLDLength": len(tld_info.suffix),
        "NoOfSubDomain": parsed_url.netloc.count("."),
        "HasObfuscation": "@" in url or "//" in url[7:],
        "NoOfObfuscatedChar": sum(c in "@[]{}|\\`^" for c in url),
        "ObfuscationRatio": sum(c in "@[]{}|\\`^" for c in url) / len(url),
        "NoOfLettersInURL": sum(c.isalpha() for c in url),
        "LetterRatioInURL": sum(c.isalpha() for c in url) / len(url),
        "NoOfDegitsInURL": sum(c.isdigit() for c in url),
        "DegitRatioInURL": sum(c.isdigit() for c in url) / len(url),
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
        "NoOfOtherSpecialCharsInURL": sum(not c.isalnum() for c in url),
        "SpacialCharRatioInURL": sum(not c.isalnum() for c in url) / len(url),
        "IsHTTPS": url.startswith("https"),
        "LineOfCode": 0,  # Placeholder
        "LargestLineLength": 0,  # Placeholder
        "HasTitle": 0,  # Placeholder
        "Title": 0,  # Placeholder
        "DomainTitleMatchScore": 0,  # Placeholder
        "URLTitleMatchScore": 0,  # Placeholder
        "HasFavicon": 0,  # Placeholder
        "Robots": 0,  # Placeholder
        "IsResponsive": 0,  # Placeholder
        "NoOfURLRedirect": 0,  # Placeholder
        "NoOfSelfRedirect": 0,  # Placeholder
        "HasDescription": 0,  # Placeholder
        "NoOfPopup": 0,  # Placeholder
        "NoOfiFrame": 0,  # Placeholder
        "HasExternalFormSubmit": 0,  # Placeholder
        "HasSocialNet": 0,  # Placeholder
        "HasSubmitButton": 0,  # Placeholder
        "HasHiddenFields": 0,  # Placeholder
        "HasPasswordField": 0,  # Placeholder
        "Bank": 0,  # Placeholder
        "Pay": 0,  # Placeholder
        "Crypto": 0,  # Placeholder
        "HasCopyrightInfo": 0,  # Placeholder
        "NoOfImage": 0,  # Placeholder
        "NoOfCSS": 0,  # Placeholder
        "NoOfJS": 0,  # Placeholder
        "NoOfSelfRef": 0,  # Placeholder
        "NoOfEmptyRef": 0,  # Placeholder
        "NoOfExternalRef": 0,  # Placeholder,
    }

    df = pd.DataFrame([features])
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df


# DeepSeek API Query
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

    return response.json()["choices"][0]["message"]["content"].strip()


# Parse DeepSeek response
def parse_deepseek_response(response_text):
    try:
        status, confidence, reason = response_text.split("|")
        status = status.strip()
        confidence = int(confidence.strip())
        reason = reason.strip()

        is_phishing = status.lower() == "phishing"

        return {"confidence": confidence, "isPhishing": is_phishing, "reason": reason}
    except Exception:
        return {
            "confidence": 5,
            "isPhishing": True,
            "reason": "DeepSeek response parsing failed.",
        }


@app.route("/")
def home():
    return render_template("index.html")  # Serves index.html


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    # Model Prediction
    features_df = extract_features(url)

    if features_df.shape[1] != model.input_shape[1]:
        return (
            jsonify(
                {
                    "error": f"Model expects {model.input_shape[1]} features, got {features_df.shape[1]}"
                }
            ),
            400,
        )

    model_prediction = model.predict(features_df)[0][0]
    model_confidence = round(model_prediction * 10, 2)
    model_is_phishing = model_prediction >= 0.5

    # DeepSeek Verification
    try:
        deepseek_response = query_deepseek(url)
        deepseek_result = parse_deepseek_response(deepseek_response)
        return jsonify(deepseek_result)
    except Exception as e:
        # Fallback to Model if DeepSeek fails
        return jsonify(
            {
                "confidence": model_confidence,
                "isPhishing": model_is_phishing,
                "reason": "DeepSeek API unavailable, using model prediction.",
            }
        )


if __name__ == "__main__":
    app.run(debug=True)
