def evaluate_retrieval(retrieved_ids, ground_truth_ids):
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(ground_truth_ids)

    true_positives = len(retrieved_set & ground_truth_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(ground_truth_set) if ground_truth_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_relation_extraction(extracted_relations, expected_relations):
    correct = 0
    for expected in expected_relations:
        if expected in extracted_relations:
            correct += 1
    accuracy = correct / len(expected_relations) if expected_relations else 0
    return accuracy
