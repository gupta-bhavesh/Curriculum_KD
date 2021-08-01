import numpy as np
import cv2

def max_contour(img):
    norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)

    thresholded = cv2.threshold(norm_image,127,255,cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    output = np.zeros_like(img)
    if len(contours) != 0:
        c = max(contours, key = cv2.contourArea)
        cv2.drawContours(output, [c], -1, 255, -1)

    return output

def dice_coef(y_true, y_pred):

    intersection = np.sum(np.logical_and(y_true, y_pred))
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def IOU(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    
    iou_score = np.sum(intersection) / np.sum(union)

    if np.isnan(iou_score)==True:
        iou_score=1

    return iou_score

def recall(y_true, y_pred):

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