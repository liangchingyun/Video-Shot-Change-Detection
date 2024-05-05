#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:07:47 2024

@author: joanneliang
"""

climate_ground = [93, 157, 232, 314, 355, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 886, 887, 1021, 1237, 1401, 1555]
news_ground = [73, 235, 301, 370, 452, 861, 1281]
ngc_ground = [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 285, 340, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 456, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 683, 703, 722, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1038, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059]


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

def Load_Frames(folder_path):
    
    video_frames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    try:
        video_frames.remove('%s/.DS_Store' % (folder_path))
    except ValueError:
        pass
    return video_frames
    

def Histogram(frames):
    t = 0
    imgs = []

    for f in frames:
        img = cv2.imread(f)
        img_resize = cv2.resize(img, (256,256)) #調整大小256*256
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY) #轉為灰階
        
        
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================='''
        
        hist = cv2.calcHist([img_gray], [0], None, [64], [1, 256])
        hist_norm = cv2.normalize(hist, hist, 1, cv2.NORM_MINMAX)
        vec = np.reshape(hist_norm, [-1]) #展平為一維向量
        imgs.append(vec) 

        t += 1
        print('\r' + '[Histogram Progress]:|%s%s|%.2f%%;' % ('█' * int(t * 20 / len(frames)), ' ' * (20 - int(t * 20 / len(frames))), float(t / len(frames) * 100)), end='')

    imgs= np.asarray(imgs, np.float32)
    return imgs


def Region_Histogram(frames):
    t = 0
    imgs = {}
    for i in range(16):
        imgs[i] = []    

    for f in frames:
        img = cv2.imread(f)
        img_resize = cv2.resize(img, (256,256)) #調整大小256*256
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY) #轉為灰階
        split = [img_gray[x:x+64,y:y+64] for x in range(0,256,64) for y in range(0,256,64)]
        
        '''===============================
        影像處理的技巧可以放這邊，來增強影像的品質
        
        ==============================='''
        i = 0
        for region in split:
            hist = cv2.calcHist([region], [0], None, [64], [1, 256])
            hist_norm = cv2.normalize(hist, hist, 1, cv2.NORM_MINMAX)
            vec = np.reshape(hist_norm, [-1]) # 展平為一維向量
            imgs[i].append(vec) 
            i += 1
            
        t += 1
        print('\r' + '[Region Histogram Progress]:|%s%s|%.2f%%;' % ('█' * int(t * 20 / len(frames)), ' ' * (20 - int(t * 20 / len(frames))), float(t / len(frames) * 100)), end='')

    for i in range(16):
        imgs[i] = np.asarray(imgs[i], np.float32)

    return imgs



def Label(frames, changed_frames):
    label = np.zeros(len(frames))
    for i in changed_frames:
        label[i-1] = 1
    return label


def PR_curve(pred, ground):
    precision, recall, _ = precision_recall_curve(ground, pred)
    
    return precision, recall

frames = Load_Frames('climate_out')
ground_label = Label(frames, climate_ground)

# Historam

shot_change_histogram = []
histogram = Histogram(frames)
metric_histogram = []

for i in range(len(histogram)-1):
    metric_val_histogram = cv2.compareHist(histogram[i], histogram[i+1], cv2.HISTCMP_BHATTACHARYYA)
    print(metric_val_histogram)
    metric_histogram.append(metric_val_histogram)
    
threshold_histogram = 0.18

for i in range(len(metric_histogram)):
    if metric_histogram[i] > threshold_histogram:
        shot_change_histogram.append(i+1)

    
#pred_label_histogram = Label(frames, shot_change_histogram)
#precision_histogram, recall_histogram, thresholds_histogram = PR_curve(pred_label_histogram, ground_label)





'''# Histogram

histogram = Histogram(frames)
pred_histogram = [0]

for i in range(len(histogram)-1):
    Bhattacharyya_histogram = cv2.compareHist(histogram[i], histogram[i+1], cv2.HISTCMP_BHATTACHARYYA) #越小越相近
    
    pred_histogram.append(Bhattacharyya_histogram)
    
precision_histogram, recall_histogram = PR_curve(pred_histogram, ground_label)'''


'''# Region Histogram
shot_change_region_histogram_temp = []
region_histogram = Region_Histogram(frames)

pred_region_histogram = {}
for i in range(16):
    pred_region_histogram[i] = []    

for i in range(16):
    region = region_histogram[i]
    for j in range(len(region)-1):
        Bhattacharyya_region_histogram = cv2.compareHist(region[j], region[j+1], cv2.HISTCMP_BHATTACHARYYA) #越小越相近
        pred_region_histogram[i].append(Bhattacharyya_region_histogram)
        
    #threshold_region_histogram = np.mean(metric_region_histogram) + np.std(metric_region_histogram)

    for i in range(len(metric_region_histogram)):
        if metric_region_histogram[i] > threshold_region_histogram:
            shot_change_region_histogram_temp.append(i+1)

shot_change_region_histogram = []
for change in shot_change_region_histogram_temp:
    if change not in shot_change_region_histogram:
        shot_change_region_histogram.append(change)

#pred_label_region_histogram = Label(frames, shot_change_region_histogram)
precision_region_histogram, recall_region_histogram = PR_curve(pred_label_region_histogram, ground_label)
'''

'''plt.plot(recall_histogram, precision_histogram, label='Histogram')
plt.plot(recall_region_histogram, precision_region_histogram, label='Region Histogram')
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
plt.title("PR Curve for Climate");'''
