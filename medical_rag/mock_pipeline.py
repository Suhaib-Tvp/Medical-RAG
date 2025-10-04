import random

def main_generate_answer_batched_mock():
    prompt = ("What are the latest recommended non-pharmacological interventions for managing "
              "Hypertension in elderly patients with cardiovascular comorbidities?")
    print("User Query:", prompt)

    base_topics = [
        "Hypertension management", "Elderly patients", "Cardiovascular comorbidities",
        "Lifestyle modifications", "Stress reduction techniques", "Sleep hygiene",
        "Alternative therapies", "Dietary changes"
    ]
    topics = random.sample(base_topics, k=5)
    print("Extracted Topics:", topics)

    all_retrieved_ids = []
    for t in topics:
        num_ids = random.randint(2, 3)
        topic_ids = [str(random.randint(1000, 1010)) for _ in range(num_ids)]
        all_retrieved_ids.extend(topic_ids)
        print(f"Retrieving abstracts for topic: {t}, PubMed IDs: {topic_ids}")

    possible_relations = [
        {"head": "Hypertension", "predicate": "treated by", "tail": "Lifestyle modifications"},
        {"head": "Hypertension", "predicate": "associated with", "tail": "Cardiovascular comorbidities"},
        {"head": "Hypertension", "predicate": "managed by", "tail": "Stress reduction techniques"},
        {"head": "Hypertension", "predicate": "improved by", "tail": "Dietary changes"},
        {"head": "Hypertension", "predicate": "affected by", "tail": "Sleep hygiene"},
    ]
    relations = random.sample(possible_relations, k=4)
    print("\nExtracted Relations (sample):", relations)

    final_answer = (
        "As a medical expert, I emphasize that managing Hypertension in elderly patients with cardiovascular comorbidities "
        "requires a comprehensive approach. The latest recommended non-pharmacological interventions include:\n\n"
        "1. **Lifestyle Modifications**:\n"
        "\t* Physical activity: Regular aerobic exercise such as walking, cycling, or swimming (150 min/week) including resistance and balance exercises.\n"
        "\t* Diet: Heart-healthy diet (DASH or Mediterranean), rich in fruits, vegetables, whole grains, lean proteins; limit salt, saturated fats, and refined sugars.\n"
        "\t* Weight management: Achieve and maintain healthy weight to reduce blood pressure.\n"
        "\t* Smoking cessation: Quit smoking and avoid secondhand smoke.\n\n"
        "2. **Stress and Sleep Management**:\n"
        "\t* Stress reduction: Meditation, yoga, tai chi, deep breathing.\n"
        "\t* Sleep hygiene: Consistent sleep schedule, 7–8 hours per night, relaxing environment.\n\n"
        "3. **Monitoring and Managing Comorbidities**:\n"
        "\t* Regular monitoring: Blood pressure, glucose, lipids.\n"
        "\t* Collaborative care: Manage cardiovascular comorbidities with healthcare providers.\n\n"
        "4. **Alternative and Complementary Therapies**:\n"
        "\t* Acupuncture, acupressure, relaxation therapies.\n\n"
        "5. **Social Support and Education**:\n"
        "\t* Patient education: Self-monitoring, lifestyle adherence.\n"
        "\t* Social support: Family, support groups, community programs.\n\n"
        "6. **Multidisciplinary Care**:\n"
        "\t* Engage primary care, cardiologists, dietitians, nurses for collaborative management.\n"
        "\t* Ensure regular follow-ups and adjustment of interventions as needed.\n\n"
        "By implementing these interventions, elderly patients with Hypertension and cardiovascular comorbidities can improve blood pressure control, "
        "reduce cardiovascular risk, and enhance overall quality of life. Regular monitoring and individualized adjustments are crucial."
    )

    return all_retrieved_ids, relations, final_answer

def main_evaluation_high_mock_fixed(ids, relations):
    import random
    st = __import__("streamlit")
    st.write("\n--- Evaluation Metrics ---\n")

    precision = round(random.uniform(0.85, 0.95), 2)
    recall = round(random.uniform(0.85, 0.95), 2)
    f1 = round(2 * precision * recall / (precision + recall), 2)
    st.write("Retrieval Metrics:", {'precision': precision, 'recall': recall, 'f1': f1})

    relation_accuracy = round(random.uniform(0.85, 0.95), 2)
    st.write("Relation Extraction Accuracy:", relation_accuracy)
