import numpy as np
import matplotlib.pyplot as plt


def calculate_tpr_fpr(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    # Convert to boolean for safe logical operations
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)

    tp = np.sum(y_true_bool & y_pred_bool)
    fp = np.sum(~y_true_bool & y_pred_bool)
    tn = np.sum(~y_true_bool & ~y_pred_bool)
    fn = np.sum(y_true_bool & ~y_pred_bool)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return tpr, fpr


#y_true = np.array([1, 0, 0, 0, 1, 0, 1, 0])
#y_true = np.array([0, 1, 0, 1, 1, 0, 0, 0])
#y_true = np.array([1, 0, 0, 1, 0, 0, 0, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
y_prob = np.array([0.01, 0.19, 0.25, 0.31, 0.41, 0.42, 0.7, 0.89])

thresholds = np.linspace(0, 1, 100)
tpr_values, fpr_values = zip(*[calculate_tpr_fpr(y_true, y_prob, threshold) for threshold in thresholds])

plt.figure(figsize=(8, 6))
plt.plot(fpr_values, tpr_values, color='darkorange', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])  # Adjusted for consistency in range
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xticks(np.arange(0, 1.1, 0.1))  # Setting x ticks from 0 to 1 with 0.1 increment
plt.yticks(np.arange(0, 1.1, 0.1))  # Setting y ticks from 0 to 1 with 0.1 increment
plt.legend(loc='lower right')
plt.grid(True)
plt.show()