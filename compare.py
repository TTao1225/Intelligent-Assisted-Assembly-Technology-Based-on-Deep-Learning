import numpy as np
import cv2
import os


def compare(filename):
    filename_less = filename+'-H-label.png'
    # filename_template = 'board-1.png'
    filename_template = 'board-2-H-label.png'
    show_result = filename+"-H.jpg"
    image_name_less = os.path.join('static/images', filename_less)
    image_name_template = os.path.join('static/images', filename_template)
    image_name_show_result = os.path.join('static/images', show_result)
    imageA = cv2.imread(image_name_less)
    imageB = cv2.imread(image_name_template)
    show_result = cv2.imread(image_name_show_result)
    ious = []
    # Ignore IoU for background class
    for cls in range(1,5):

        pred_inds = imageA == cls

        target_inds = imageB == cls

        intersection = (pred_inds[target_inds]).sum().item()  # Cast to long to prevent overflows
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        if union == 0:
            ious.append("error")  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / max(union, 1))

    print(ious)
    status = 0
    for i, iou in enumerate(ious):
        pred_inds = imageA == i+1
        target_inds = imageB == i+1
        if iou > 0.4:
            show_result[:,:,1][pred_inds[:,:,1]] = 255
        else:
            status += 1
            show_result[:,:,2][target_inds[:,:,1]] = 255


    cv2.imwrite("./static/images/" + filename + '-com.jpg', show_result)
    if status > 0:
        return "bad"
    else:
        return "good"