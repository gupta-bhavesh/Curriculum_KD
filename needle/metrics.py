import numpy as np
import cv2

def max_contour(inp_img):

    norm_image = cv2.normalize(inp_img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)

    thresholded = cv2.threshold(norm_image,25,255,cv2.THRESH_BINARY)[1]

    contours,_ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max = [0,0,0,0]

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)

        if w*h > max[2]*max[3]:
            max = [x,y,w,h]
    point1 = (max[0],max[1]+max[3])
    point2 = (max[0]+max[2],max[1])

    angle = np.arctan(max[3]/max[2])
  
    return point1, point2, angle

def pred_needle_img(point1, point2, size=(440, 500)):

    zeros = np.zeros(size)
    cv2.line(zeros, point1, point2, (255,255,255), 2)
    
    return zeros

def inline_BB(point1, point2, shape, a = 15):
    
    x1, y1 = point1
    x2, y2 = point2

    zeros = np.zeros(shape)
    cv2.line(zeros, (x1, y1), (x2, y2), (255,255,255), 1)

    tan = -1*(y1-y2)/(x1-x2)
    theta = np.arctan(tan)

    cos_minus_sin = np.cos(theta) - np.sin(theta)

    cos_plus_sin = np.cos(theta) + np.sin(theta)

    point1 = [x1-a*cos_minus_sin, y1 + a*cos_plus_sin]
    point2 = [x1-a*cos_plus_sin, y1 - a*cos_minus_sin]
    point3 = [x2 + a*cos_minus_sin, y2 - a*cos_plus_sin]
    point4 = [x2 + a*cos_plus_sin, y2 + a*cos_minus_sin]

    pts = np.array([point1, point2, 
                    point3, point4],
                   np.int32)

    image = cv2.polylines(zeros, [pts], True, (255,255,255), 1)
    return image

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

def angle_acc(act_angle, pred_angle):
    return (act_angle-pred_angle)**2

def dist_acc(pred_point1, pred_point2, act_point1, act_point2, size):

    center = np.array(size)/2

    pred_point1 = np.array(pred_point1)
    pred_point2 = np.array(pred_point2)
    
    pred_dist = np.cross(pred_point2-pred_point1,center-pred_point1)/np.linalg.norm(pred_point2-pred_point1)

    act_point1 = np.array(act_point1)
    act_point2 = np.array(act_point2)
    
    act_dist = np.cross(act_point2-act_point1,center-act_point1)/np.linalg.norm(act_point2-act_point1)
    
    return np.square(act_dist-pred_dist)

def dice_coef(y_true, y_pred):

    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def IOU(act_rect, pred_rect):

    intersection = np.logical_and(act_rect, pred_rect)
    union = np.logical_or(act_rect, pred_rect)
    
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score

def recall(act_rect, pred_rect):

    tp = np.sum(np.logical_and(act_rect, pred_rect))
    fn = np.sum(np.logical_and(act_rect, 1-pred_rect))
    recall = tp/(tp+fn)

    return recall

def precision(act_rect, pred_rect):

    tp = np.sum(np.logical_and(act_rect, pred_rect))
    fp = np.sum(np.logical_and(1-act_rect, pred_rect))
    precision = tp/(tp+fp)

    return precision
    
def F1(precision,recall):
    F1=2*precision*recall/(precision+recall)
    
    return F1