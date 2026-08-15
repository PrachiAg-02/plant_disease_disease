import os
import json
from google import genai
from google.genai import types

# Expert treatments for the trained disease classes
TREATMENT_DB = {
    "angular_leaf_spot": {
        "disease_name": "Angular Leaf Spot (Phaeoisariopsis griseola)",
        "chemical": "Apply copper-based fungicides or Mancozeb at 2.5g/L water at the first sign of angular lesions.",
        "organic": "Foliar spray of 5% Neem seed oil extract or Bacillus subtilis bio-fungicide. Remove and destroy infected lower leaves.",
        "prevention": "Rotate crops with non-legumes for at least 2 years, avoid overhead irrigation, and use certified pathogen-free seeds."
    },
    "bean_rust": {
        "disease_name": "Bean Rust (Uromyces appendiculatus)",
        "chemical": "Spray Triazole-based or Chlorothalonil fungicides every 10–14 days during high humidity periods.",
        "organic": "Dust with wettable sulfur powder or spray diluted potassium bicarbonate solutions early in the morning.",
        "prevention": "Increase row spacing to improve air flow and plant rust-resistant crop varieties."
    },
    "healthy": {
        "disease_name": "Healthy Plant Foliage",
        "chemical": "No chemical treatments necessary.",
        "organic": "Maintain regular organic compost application and balanced soil nutrient levels.",
        "prevention": "Continue standard crop scouting, proper watering schedules, and monitor for early pest infestation."
    }
}

def get_treatment_recommendation(disease_name: str, confidence: float) -> dict:
    # 1. Local database lookup
    if disease_name in TREATMENT_DB:
        record = TREATMENT_DB[disease_name]
        return {
            "disease": record["disease_name"],
            "raw_class": disease_name,
            "confidence": round(confidence * 100, 2),
            "source": "Local Agronomy DB",
            "treatment": {
                "chemical": record["chemical"],
                "organic": record["organic"],
                "prevention": record["prevention"]
            }
        }

    # 2. Dynamic generation using Gemini API
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "disease": disease_name,
            "raw_class": disease_name,
            "confidence": round(confidence * 100, 2),
            "source": "Fallback (No API Key)",
            "treatment": {
                "chemical": "Consult your local agricultural extension service.",
                "organic": "Isolate the affected plant canopy immediately.",
                "prevention": "Avoid water splash on foliage and ensure soil drainage."
            }
        }

    client = genai.Client(api_key=api_key)
    prompt = f"""
    A plant pathology deep learning model diagnosed: '{disease_name}' with {confidence*100:.1f}% confidence.
    Respond strictly in valid JSON format matching this schema:
    {{
        "chemical": "string (specific active ingredients and application dosage)",
        "organic": "string (eco-friendly remedies and bio-controls)",
        "prevention": "string (field management and cultural prevention practices)"
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        return {
            "disease": disease_name,
            "raw_class": disease_name,
            "confidence": round(confidence * 100, 2),
            "source": "Gemini AI Engine",
            "treatment": data
        }
    except Exception as e:
        return {
            "disease": disease_name,
            "raw_class": disease_name,
            "confidence": round(confidence * 100, 2),
            "source": "Fallback Error",
            "error": str(e),
            "treatment": {
                "chemical": "Isolate affected crops and apply broad-spectrum fungicide.",
                "organic": "Apply neem extract spray.",
                "prevention": "Maintain clean equipment and rotate planting beds."
            }
        }