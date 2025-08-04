import pandas as pd

submission_df = pd.read_csv("v3_bert_cnn_submission.csv")
ground_truth_df = pd.read_csv("shopee-product-matching/train.csv")

gt_dict = ground_truth_df.groupby('label_group')['posting_id'].apply(list).to_dict()

true_labels = {
    pid: set(gt_dict[label]) for pid, label in zip(ground_truth_df['posting_id'], ground_truth_df['label_group'])
}

pred_labels = {
    row['posting_id']: set(row['matches'].split())
    for _, row in submission_df.iterrows()
}

pred_labels_top50 = {
    row['posting_id']: set(row['matches'].split()[:50])
    for _, row in submission_df.iterrows()
}

# F1 score
def f1_score_row(pred_set, true_set):
    intersection = len(pred_set & true_set)
    if intersection == 0:
        return 0.0
    precision = intersection / len(pred_set)
    recall = intersection / len(true_set)
    return 2 * (precision * recall) / (precision + recall)

# Recall@50
def f1_score_at_50(pred_set, true_set):
    intersection = len(pred_set & true_set)
    if intersection == 0:
        return 0.0
    precision = intersection / len(pred_set)  # usually 50
    recall = intersection / len(true_set)
    return 2 * (precision * recall) / (precision + recall)
    
    
# === Calculate metrics ===
f1_scores = []
recall_scores = []

for pid in submission_df['posting_id']:
    pred_set_full = pred_labels[pid]
    pred_set_top50 = pred_labels_top50[pid]
    true_set = true_labels[pid]

    f1_scores.append(f1_score_row(pred_set_full, true_set))
    recall_scores.append(f1_score_at_50(pred_set_top50, true_set))

mean_f1 = sum(f1_scores) / len(f1_scores)
mean_recall_50 = sum(recall_scores) / len(recall_scores)

print(f"Mean F1 score: {mean_f1:.5f}")
print(f"Mean Recall@50: {mean_recall_50:.5f}")