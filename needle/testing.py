import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils import max_contour
from metrics import dice_coef, IOU, recall, precision, F1
import torch.nn.functional as F

def visualizing_results(model, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():

        figure, axes = plt.subplots(nrows=num_images, ncols=4, figsize=(15,3.75*num_images))

        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)            

            for j in range(inputs.size()[0]):
                images_so_far += 1
                img = inputs.cpu().data[j].squeeze()
                label = labels.cpu().data[j].squeeze()
                label = label.squeeze().cpu().numpy()
                pred = outputs.cpu().data[j].squeeze()
                pred = nn.Softmax()(pred)[1]
                pred = pred.squeeze().cpu().numpy()

                axes[j, 0].imshow(np.transpose(img, (1,2,0)), cmap='gray')
                axes[j, 1].imshow(pred, cmap='gray')
                pred[pred>0.5] = 255
                pred[pred<=0.5] = 0
                # post_process = max_contour(pred)
                axes[j, 2].imshow(pred, cmap='gray')
                axes[j, 3].imshow(label, cmap='gray')
                cols = ['Input', 'Prediction', 'Post-Process', 'Ground Truth']

                for ax, col in zip(axes[0], cols):
                    ax.set_title(col)

                if images_so_far == num_images:

                    model.train(mode=was_training)
                    figure.tight_layout()
                    return
        model.train(mode=was_training)

def testing(model, dataloaders):
    was_training = model.training
    model.eval()

    dist_acc_arr = []
    angle_acc_arr = []
    dice_arr = []
    iou_arr = []
    precision_arr = []
    recall_arr = []
    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)            

            for j in range(inputs.size()[0]):
                
                img = inputs.cpu().data[j].squeeze()
                label = labels.cpu().data[j].squeeze()
                label = label.squeeze().cpu().numpy()
                pred = outputs.cpu().data[j].squeeze()
                pred = F.softmax(pred, dim=0)[1]
                pred = pred.squeeze().cpu().numpy()

                pred_point1, pred_point2, pred_angle = max_contour(pred)
                act_point1, act_point2, act_angle = max_contour(label)

                pred_rect = inline_BB(pred_point1, pred_point2, pred.shape)/255
                act_rect = inline_BB(act_point1, act_point2, pred.shape)/255

                post_processing = pred_needle_img(pred_point1, pred_point2, size = pred.shape)

                dist_acc_arr.append(dist_acc(pred_point1, pred_point2, act_point1, act_point2, pred.shape))
                angle_acc_arr.append(angle_acc(pred_angle, act_angle))
                dice_arr.append(dice_coef(label, post_processing/255))
                iou_arr.append(IOU(act_rect, pred_rect))
                precision_arr.append(precision(act_rect, pred_rect))
                recall_arr.append(recall(act_rect, pred_rect))

        print("-"*10 + "Distance Error"+ "-"*10)
        print(np.sqrt(np.mean(dist_acc_arr)))
        
        print("-"*10 + "Angle Error"+ "-"*10)
        print(np.sqrt(np.mean(angle_acc_arr)))
        
        print("-"*10 + "Max Dice"+ "-"*10)
        print(max(dice_arr))
        
        print("-"*10 + "Avg Dice"+ "-"*10)
        print(np.mean(dice_arr))
        
        print("-"*10 + "Avg IOU"+ "-"*10)
        print(np.mean(iou_arr))
        
        print("-"*10 + "Avg Precision"+ "-"*10)
        print(np.mean(precision_arr))
        
        print("-"*10 + "Avg Recall"+ "-"*10)
        print(np.mean(recall_arr))
        
        print("-"*10 + "F1 Score"+ "-"*10)
        print(F1(np.mean(precision_arr), np.mean(recall_arr)))
        
        model.train(mode=was_training)
    return np.sqrt(np.mean(dist_acc_arr)), np.sqrt(np.mean(angle_acc_arr)), max(dice_arr), np.mean(dice_arr), np.mean(iou_arr), np.mean(precision_arr), np.mean(recall_arr), F1(np.mean(precision_arr), np.mean(recall_arr))