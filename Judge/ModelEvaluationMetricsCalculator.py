import json

# Load the JSON file
file_path = "YOUR-FILE-PATH"
with open(file_path, 'r', encoding="utf-8") as file:
    data = json.load(file)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, matthews_corrcoef

# Extract labels and predictions
labels = [entry['label'] for entry in data]
predictions = [entry['predict_label'] for entry in data]

# Calculate Precision, Recall, Accuracy, F1 Score, and Matthews Correlation Coefficient
precision = precision_score(labels, predictions)
recall = recall_score(labels, predictions)
accuracy = accuracy_score(labels, predictions)
f1 = f1_score(labels, predictions)
mcc = matthews_corrcoef(labels, predictions)

print(precision)
print(recall)
print(accuracy)
print(f1)
print(mcc)
