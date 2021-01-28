import numpy as np
from collections import Counter


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

def weighted_precision(y_true, y_pred):
    """
   Function to calculate weighted averaged precision
   :param y_true: list of true values
   :param y_pred: list of predicted values
   :return: weighted precision score
   """

   # find number of classes
    num_classes = len(np.unique(y_true))

   # create class:sample count dictionary
   # it looks something like this
   # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)

   # initialize precision to zero
    precision = 0

   # loop over all classes
    for class_ in range(num_classes):
       # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

       # calculate tp and fp for class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)

       # calculate precision of class
        temp_precision = tp / (tp + fp)

       # multiply precision with count of class
        weighted_precision = class_counts[class_] * temp_precision

       # add to overall precision
        precision += weighted_precision
    # calculate overall precision by dividing by
    # total number of samples
    overall_precision = precision / len(y_true)

    return overall_precision


# weighted f1-score

def weighted_f1(y_true, y_pred):
    """
   Function to calculate weighted averaged f1 score
   :param y_true: list of true values
   :param y_pred: list of predicted values
   :return: weighted f1 score
   """

   # number of classes
    num_classes = len(np.unique(y_true))

    # create class:sample count dictionary
    class_counts = Counter(y_true)

    # initialize f1 to 0
    f1 = 0

    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]

        # precision and recall for class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)

        # calculate f1 of class
        if p + r != 0:
            temp_f1 = 2 * p * r / (p + r)
        else:
            temp_f1 = 0

        # multiply f1 with count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1

        # add to f1 precision
        f1 += weighted_f1

        # calculate overall F1 by dividing by total
        # number of samples
    overall_f1 = f1 / len(y_true)
    return overall_f1


# precision at k (P@k)

def pk(y_true, y_pred, k):
    """
   Function to calculate precision at k
   :param y_true: list of values, actual classes
   :param y_pred: list of values, predicted classes
   :param k: the value for k
   :return: precision at given value k
   """

    # we are only interested in top-k predictions
    y_pred = y_pred[:k]
    # convert predictions to set
    pred_set = set(y_pred)
    # convert actual values to set
    true_set = set(y_true)
    # find common values
    common_values = pred_set.intersection(true_set)
    # return length of common values over k
    return len(common_values) / len(y_pred[:k])

# average precision at k or AP@k


def apk(y_true, y_pred, k):
    """
    This function calculate average precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: average precision at a given value k
    """

    # initialize p@k list of values
    pk_values = []
    # loop over all k, from 1 to k + 1
    for i in range(1, k+1):
        # calculate p@i and append to list
        pk_values.append(pk(y_true, y_pred, i))
    # if we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # else, we return the sum of list over length of list
    return sum(pk_values) / len(pk_values)

# mean average precision at k (MAP@K)


def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """
    # initialize empty list for apk values
    apk_values = []
    # loop over all samples
    for i in range(len(y_true)):
        # store apk values for every sample
        apk_values.append(

            apk(y_true[i], y_pred[i], k=k)
        )
    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

# ===================================================================================

# METRICS FOR REGRESSION


def mean_absolute_error(y_true, y_pred):
    """
    This function calculates mae
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """

    # initialize error to zero
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate absolute error
        error += np.abs(yt - yp)

    # return mean error
    return error / len(y_true)


def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared error
    """

    # initialize error to zero
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        error += np.abs(yt - yp) ** 2

    # return mean error
    return error / len(y_true)


# mean squared log error

def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates msle
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared log error
    """

    # initialize error to zero
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2

    # return mean error
    return error / len(y_true)


def mean_percentage_error(y_true, y_pred):
    """
    This function calculates mpe
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += (yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)


def mean_abs_percentage_error(y_true, y_pred):
    """
    This function calculates MAPE
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += np.abs(yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)


def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """

    # calculate the mean value of true values
    mean_true_value = np.mean(y_true)

    # initialize numerator with 0
    numerator = 0
    # initialize denominator with 0
    denominator = 0

    # loop over all true and predicted values
    for yt, yp in zip(y_true, y_pred):
        # update numerator
        numerator += (yt - yp) ** 2
        # update denominator
        denominator += (yt - mean_true_value) ** 2
        # calculate the ratio
    ratio = numerator / denominator
    # return 1 - ratio
    return (1 - ratio)


def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient
    for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    numerator = (tp * tn) - (fp * fn)
    denominator = (
    (tp + fp) *
    (fn + tn) *
    (fp + tn) *
    (tp + fn)
    )
    denominator = denominator ** 0.5
    return numerator/denominator