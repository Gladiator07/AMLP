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

# function to calculate true positives

def true_positive(y_true, y_pred):
    """
    Function to calculate true positives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true positives
    """
    # initialize counter
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==1 and yp==1:
            tp += 1
    return tp


# function to calculate true negatives

def true_negative(y_true, y_pred):
    """
    Function to calculate true negatives
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of true negatives
    """
    # intialize the counter
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==0 and yp==0:
            tn += 1
    return tn


# function to calculate false positives

def false_positive(y_true, y_pred):
    """
    Function to calculate false positive
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives
    """
    # intialize the counter
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==0 and yp==1:
            fp += 1
    return fp


# function to calculate false negatives

def false_negative(y_true, y_pred):
    """
    Function to calculate false negative
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: number of false positives
    """
    # intialize the counter
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt==1 and yp==0:
            fn += 1
    return fn

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


# function to calculate precision

def precision(y_true, y_pred):
    """
    Function to calculate precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return : precision score
    """
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    precision = tp / (tp + fp)
    return precision

# function to calculate recall

def recall(y_true, y_pred):
    """
    Function to calculate recall
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return : recall score
    """
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp + fn)
    return recall

def f1(y_true, y_pred):
    """
    Function to calculate f1 score
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return : f1 score
    """

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    score = 2 * p * r / (p + r)
    return score