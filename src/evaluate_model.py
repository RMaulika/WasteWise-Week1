import numpy as np, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(model, test_gen, class_indices, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    preds = model.predict(test_gen, verbose=1)
    y_pred, y_true = np.argmax(preds, axis=1), test_gen.classes
    labels = list(class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
