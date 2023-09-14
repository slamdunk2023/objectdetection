import numpy as np
import cv2 as cv
import torch

def apply_nms(boxes, scores, threshold=0.5):
    """
    Apply Non-Maximum Suppression.
    
    Args:
    - boxes (list or tensor): List of predicted bounding boxes.
    - scores (list or tensor): List of scores for each bounding box.
    - threshold (float): Overlap threshold for NMS.
    
    Returns:
    - List of indices of boxes to keep.
    """
    
    # Check if boxes and scores are lists or tensors, then convert accordingly
    if isinstance(boxes, list):
        boxes_array = np.array([item.detach().numpy() for item in boxes])
    else:
        boxes_array = boxes.detach().numpy()
        
    if isinstance(scores, list):
        scores_array = np.array([item.detach().numpy() for item in scores])
    else:
        scores_array = scores.detach().numpy()
    
    # Apply NMS and get indices of boxes to keep
    indices = cv.dnn.NMSBoxes(boxes_array.tolist(), scores_array.tolist(), score_threshold=0.1, nms_threshold=threshold)

    # Return indices directly
    return [i for i in indices]
