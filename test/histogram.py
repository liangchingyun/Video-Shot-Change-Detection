#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:52:04 2024

@author: joanneliang
"""

import cv2
import numpy as np

def calculate_hist_diff(frame1, frame2):
    """
    Calculate the histogram difference between two frames.
    """
    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

def detect_shot_changes(video_frames, ground_truth_shots, threshold):
    """
    Detect shot changes in a video using histogram difference and a single threshold.
    """
    predicted_shots = []
    prev_frame = None

    for i, frame in enumerate(video_frames):
        if prev_frame is not None:
            hist_diff = calculate_hist_diff(prev_frame, frame)
            if hist_diff > threshold:
                predicted_shots.append(i)
        prev_frame = frame

    return predicted_shots

def calculate_precision_recall(predicted_shots, ground_truth_shots):
    """
    Calculate precision and recall of predicted shot changes.
    """
    true_positives = sum(1 for pred in predicted_shots if pred in ground_truth_shots)
    false_positives = sum(1 for pred in predicted_shots if pred not in ground_truth_shots)
    false_negatives = sum(1 for gt in ground_truth_shots if gt not in predicted_shots)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0

    return precision, recall

# Example usage
# Load video frames and ground truth shot change frames
video_frames = [cv2.imread('frame_{}.jpg'.format(i)) for i in range(100)]  # Replace with your frame loading logic
ground_truth_shots = [10, 25, 40, 60, 80]  # Replace with your ground truth shot change frames

# Set the threshold value
threshold = 0.5

# Detect shot changes
predicted_shots = detect_shot_changes(video_frames, ground_truth_shots, threshold)

# Calculate precision and recall
precision, recall = calculate_precision_recall(predicted_shots, ground_truth_shots)

print("Predicted shot change frames:", predicted_shots)
print("Precision:", precision)
print("Recall:", recall)