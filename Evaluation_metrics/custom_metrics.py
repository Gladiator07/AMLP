import numpy as np


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

    return correct_counter / len(y_true)

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
        if yt == 1 and yp == 1:
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
        if yt == 0 and yp == 0:
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
        if yt == 0 and yp == 1:
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
        if yt == 1 and yp == 0:
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

# true positive rate


def tpr(y_true, y_pred):
    """
    Function to calculate tpr
    :param y_true: list of true values
    "param y_pred: list of predicted values
    :return" tpr/recall
    """

    return recall(y_true, y_pred)


def fpr(y_true, y_pred):
    """
    Function to calculate fpr
    :param y_true: list of true values
    "param y_pred: list of predicted values
    :return" fpr
    """
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)

    return fp / (tn + fp)

# log-loss metric


def log_loss(y_true, y_proba):
    """
    Function to calculate log loss
    :param y_true: list of true values
    :param y_proba: list of probabilities for 1
    :return: overall log loss
    """

    # epsilon values (used to clip probabilities)
    epsilon = 1e-15

    loss = []

    for yt, yp in zip(y_true, y_proba):
        # adjust probability
        # 0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        yp = np.clip(yp, epsilon, 1-epsilon)
        # calculate loss for one sample
        temp_loss = -1.0 * (yt * np.log(yp) + (1 - yt) * np.log(1 - yp))

        loss.append(temp_loss)

    # return mean loss over all samples
    return np.mean(loss)


# macro-averaged precision

def macro_precision(y_true, y_pred):
    """
    Function to calculate macro averaged precision
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: macro precision score
    """

    # find the number of classes by taking length
    # of unique values in true list

    num_classes = len(np.unique(y_true))

    # initialze precision to zero
    precision = 0

    # loop over all classes
    for class_ in range(num_classes):

        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        tp = true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)

        # calculate precision for current class
        temp_precision = tp / (tp + fp)

        # keep adding precision for all class
        precision += temp_precision

        # calculate and return average precision over all classes

        precision /= num_classes

        return precision


# micro-precision

def micro_precision(y_true, y_pred):
    """
   Function to calculate micro averaged precision
   :param y_true: list of true values
   :param y_pred: list of predicted values
   :return: micro precision score
   """
    # find number of classes
    num_classes = len(np.unique(y_true))

    # initialize tp and fp to 0
    tp = 0
    fp = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # calculate true positive for current class
        # and update overall tp

        tp += true_positive(temp_true, temp_pred)

        # calculate false positive for current class
        # and update overall tp

        fp += false_positive(temp_true, temp_pred)

        # calculate and return overall precision
        precision = tp / (tp + fp)

        return precision


# weighted precision