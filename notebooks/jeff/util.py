from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, confusion_matrix, plot_confusion_matrix, f1_score, roc_auc_score
import numpy as np

def evaluate_model(model, X, y, plot_cf):
    '''
    Compute summary metrics for classifier models.
    Returns accuracy, precision, recall, and F1 score.
    Prints each of those metrics plus ROC score.

    Parameters
    ----------
    model: classifer model that has been train on X and y already
    X: Independent variable dataframe
    y: Target variable array
    plot_cf: boolean. If true, plots confusion matrix.

    Returns
    -------
    Prints Accuracy, Precision, Recall, F1, and ROC scores
    Returns Accuracy, Precision, Recall, and F1
    '''

    y_preds = model.predict(X)

    acc_score = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    try:
        roc = roc_auc_score(y, model.decision_function(X))
    except AttributeError:
        roc = np.nan

    print(f"Accuracy: {acc_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC: {roc:.4f}")

    if plot_cf == True:
        plot_confusion_matrix(model, X, y)


    return acc_score, precision, recall, f1


# Filter for only multicollinearity above a certain threshold
def high_corr(df, threshold):
    '''
    Find variables with a correlation above a pre-defined threshold.
    
    Parameters
    ----------
    df: DataFrame of interest for measuring correlation
    threshold: correlation threshold. Function will return pairings with a correlation above this threshold.

    Returns
    -------
    DataFrame of column pairings with a correlation above the defined threshold
    '''

    mult_corr = df.corr().abs().stack().reset_index().sort_values(0, ascending=False)

    # zip the variable name columns (Which were only named level_0 and level_1 by default) in a new column named "pairs"
    mult_corr['pairs'] = list(zip(mult_corr.level_0, mult_corr.level_1))

    # set index to pairs
    mult_corr.set_index(['pairs'], inplace = True)

    #d rop level columns
    mult_corr.drop(columns=['level_1', 'level_0'], inplace = True)

    # rename correlation column as cc rather than 0
    mult_corr.columns = ['cc']

    # drop duplicates. This could be dangerous if you have variables perfectly correlated 
    # with variables other than themselves.
    mult_corr.drop_duplicates(inplace=True)

    highly_corr = mult_corr[(mult_corr.cc>threshold)] #& (test.cc <1)]

    return highly_corr