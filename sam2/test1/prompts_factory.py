import random

NEGATIVE_PROMPT = "cartoon, low quality, blurry, text, watermark"

ANOMALY_DESCRIPTIONS = {
    "bottle": {
        "broken_large": "large broken glass area",
        "broken_small": "small glass fracture",
        "contamination": "contamination spots"
    }
}

def build_prompts(obj, anomaly):

    if obj not in ANOMALY_DESCRIPTIONS:
        desc = f"{anomaly} defect"
        return f"a photo of a {obj} with {desc}", NEGATIVE_PROMPT

    if anomaly == "combined":
        available_defects = [
            k for k in ANOMALY_DESCRIPTIONS[obj].keys()
            if k != "combined"
        ]

        if len(available_defects) >= 2:
            selected_keys = random.sample(available_defects, 2)
            desc1 = ANOMALY_DESCRIPTIONS[obj][selected_keys[0]]
            desc2 = ANOMALY_DESCRIPTIONS[obj][selected_keys[1]]
            desc = f"combined {desc1} and {desc2}"
        elif len(available_defects) == 1:
            desc = ANOMALY_DESCRIPTIONS[obj][available_defects[0]]
        else:
            desc = "multiple defects"

    else:
        if anomaly in ANOMALY_DESCRIPTIONS[obj]:
            desc = ANOMALY_DESCRIPTIONS[obj][anomaly]
        else:
            desc = f"{anomaly.replace('_', ' ')} defect"

    area_adjectives = ["huge", "massive", "covering the entire area", "widespread"]
    selected_adj = random.choice(area_adjectives)

    prompt = f"a photo of a {obj} with {selected_adj} {desc}"

    return prompt, NEGATIVE_PROMPT

if __name__ == "__main__":

    print(f"\n--- Testing Normal for Bottle ---")
    p, n = build_prompts("bottle", "broken_large")
    print(p)
    print(n)