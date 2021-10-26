from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, plot_confusion_matrix

def evaluate_model(model, X, y, plot_cf):

    y_preds = model.predict(X)

    acc_score = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)

    print(f"Accuracy: {acc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    if plot_cf == True:
        plot_confusion_matrix(model, X, y)

    return acc_score, precision, recall