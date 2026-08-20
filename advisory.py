ADVISORY_DATABASE = {
    "angular_leaf_spot": {
        "chemical": "Apply copper-based bactericides/fungicides (e.g., Copper Oxychloride 50 WP) or Mancozeb at the first onset of lesions.",
        "organic": "Spray Neem oil extract (0.5%) or potassium bicarbonate solution. Introduce biocontrol agents like *Bacillus subtilis*.",
        "prevention": "Ensure wide crop spacing for adequate airflow, avoid overhead sprinkler irrigation, and practice a 2-year crop rotation."
    },
    "bean_rust": {
        "chemical": "Spray triazole-based systemic fungicides such as Tebuconazole or Chlorothalonil before flowering.",
        "organic": "Apply wettable sulfur sprays or certified bio-fungicides (*Trichoderma harzianum*) on foliage surfaces.",
        "prevention": "Remove and incinerate infected crop residue immediately. Plant certified rust-resistant cultivar seeds."
    },
    "healthy": {
        "chemical": "No chemical intervention required.",
        "organic": "Maintain standard soil fertility with balanced organic compost and vermicompost tea.",
        "prevention": "Continue regular field scouting and maintain drip irrigation to prevent foliar dampness."
    }
}

def get_treatment_plan(disease_name: str) -> dict:
    """Returns treatment plan for a given disease class label."""
    normalized_key = disease_name.strip().lower().replace(" ", "_")
    
    if normalized_key in ADVISORY_DATABASE:
        return ADVISORY_DATABASE[normalized_key]
        
    for key, plan in ADVISORY_DATABASE.items():
        if key in normalized_key or normalized_key in key:
            return plan
            
    return {
        "chemical": "Consult local certified agronomy extension office for targeted fungicide guidelines.",
        "organic": "Apply general bio-fungicide foliar sprays (Neem oil/Bacillus subtilis).",
        "prevention": "Quarantine affected area, prune infected leaf matter, and ensure adequate spacing."
    }