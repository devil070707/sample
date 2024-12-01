# Confusion matrix values
TP = 90   # True Positive
FP = 210  # False Positive
FN = 140  # False Negative
TN = 9560 # True Negative

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Accuracy: {accuracy:.4f}")

# Calculate Error Rate
error_rate = (FP + FN) / (TP + TN + FP + FN)
print(f"Error Rate: {error_rate:.4f}")

# Calculate Precision
precision = TP / (TP + FP)
print(f"Precision: {precision:.4f}")

# Calculate Recall (Sensitivity)
recall = TP / (TP + FN)
print(f"Recall: {recall:.4f}")


# Accuracy: How many predictions (both positive and negative) are correct.
# Error Rate: How many predictions are incorrect.
# Precision: Among the predicted positives, how many are actual positives.
# Recall: Among the actual positives, how many are correctly predicted as positive.