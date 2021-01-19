from tp_fp_fn_tn import true_positive, true_negative, false_positive, false_negative

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: accuracy score
    """

    # intialize counter
    correct_counter = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct_counter += 1
    
    return correct_counter/ len(y_true)


# accuracy using TP, FP, TN, FN

def accuracy_v2(y_true, y_pred):
    """
    Function to calculate accuracy using tp,fp,fn,tn
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return : accuracy score
    """

    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    accuracy_score = (tp + tn) / (tp + fp + fn + tn)
    return accuracy_score
