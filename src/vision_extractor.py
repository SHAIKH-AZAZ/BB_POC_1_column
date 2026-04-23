import base64
import requests
import re
import time
from config import OPENAI_API_KEY


API_URL = "https://api.openai.com/v1/chat/completions"


# -----------------------------------
# ENCODE IMAGE
# -----------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -----------------------------------
# CLEAN JSON RESPONSE
# -----------------------------------
def clean_json_string(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


# -----------------------------------
# GPT EXTRACTION WITH RETRY
# -----------------------------------
def extract_from_image(image_path, prompt, retries=3):

    base64_image = encode_image(image_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "gpt-4.1-mini",
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    }

    attempt = 0

    while attempt < retries:
        try:
            print(f"🧠 Extracting → {image_path}")

            response = requests.post(
                API_URL,
                headers=headers,
                json=payload,
                timeout=300  # ✅ increased timeout
            )

            if response.status_code != 200:
                raise Exception(response.text)

            result = response.json()["choices"][0]["message"]["content"]

            return clean_json_string(result)

        except requests.exceptions.ReadTimeout:
            attempt += 1
            print(f"⚠ Timeout... Retrying ({attempt}/{retries})")
            time.sleep(5)

    raise Exception("❌ Extraction failed after retries")