import numpy as np

def dice_coef(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    intersection = np.sum(np.logical_and(y_true, y_pred))
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def IOU(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    
    iou_score = np.sum(intersection) / np.sum(union)

    if np.isnan(iou_score)==True:
        iou_score=1

    return iou_score

def recall(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum(np.logical_and(y_true, y_pred))
    fn = np.sum(np.logical_and(y_true, 1-y_pred))
    recall = tp/(tp+fn)

    if np.isnan(recall)==True:
        recall=1

    return recall

def precision(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum(np.logical_and(y_true, y_pred))
    fp = np.sum(np.logical_and(1-y_true, y_pred))
    precision = tp/(tp+fp)
    
    if np.isnan(precision)==True:
        precision=1

    return precision
    
def F1(precision,recall):
    F1=2*precision*recall/(precision+recall)
    
    return F1