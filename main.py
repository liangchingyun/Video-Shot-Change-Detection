#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:58:29 2024

@author: joanneliang
"""

climate_ground = [93, 157, 232, 314, 355, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 886, 887, 1021, 1237, 1401, 1555]
news_ground = [73, 235, 301, 370, 452, 861, 1281]
ngc_ground = [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 285, 340, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 456, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 683, 703, 722, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1038, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059]


import cv2
import os
from natsort import natsorted
import matplotlib.pyplot as plt


def Load_Frames(folder_path):
    video_frames = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    try:
        video_frames.remove('%s/.DS_Store' % (folder_path))
    except ValueError:
        pass
    video_frames = natsorted(video_frames)

    return video_frames

def Shot_Change_Detection(frames, ts, tb):
    t = 0
    pred_shot_change = []
    diffs = []
    
    diff_add = 0
    grad = []

    for i, f in enumerate(frames):
        img = cv2.imread(f)
        img_resize = cv2.resize(img, (256,256)) #調整大小256*256
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY) #轉為灰階
        hist = cv2.calcHist([img_gray], [0], None, [64], [1, 256])
        
        if i == 0:
            prev_hist = hist
            
        diff = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
        diffs.append(diff)

        
        
        if diff > tb:
            pred_shot_change.append(i)
        elif diff > ts:
            diff_add += diff
            grad.append(i)
            if diff_add > tb:
               pred_shot_change.extend(grad)
               diff_add = 0
               grad = []
        else:
            diff_add = 0
            grad = []
            
                
        prev_hist = hist
            
        t += 1
        print('\r' + '[Progress]:|%s%s|%.2f%%;' % ('█' * int(t * 20 / len(frames)), ' ' * (20 - int(t * 20 / len(frames))), float(t / len(frames) * 100)), end='')
                

    return pred_shot_change, diffs

def Precision_Recall(pred, ground):
    TP = len(set(ground).intersection(set(pred)))
    FP = len(set(pred) - set(ground))
    FN = len(set(ground) - set(pred))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall

def Remove_Consecutive_Changes(changes):
    removed = []
    prev = 0
    for i in changes:
        if i != prev + 1:
            removed.append(i)
        prev = i
    return removed


#news
news_prediction, news_diffs = Shot_Change_Detection(Load_Frames('news_out'), 0.09, 0.1)
news_prediction = Remove_Consecutive_Changes(news_prediction)
news_P , news_R = Precision_Recall(news_prediction, news_ground)
print('\n\nNews:\n  Precision: %s\n  Recall: %s\n' % (news_P, news_R))

#climate
climate_P_list , climate_R_list = [], []
for i in range(5, 55, 5):
    climate_prediction, climate_diffs = Shot_Change_Detection(Load_Frames('climate_out'), i / 1000, 0.1)
    climate_prediction = [x + 1 for x in climate_prediction]
    climate_P , climate_R = Precision_Recall(climate_prediction, climate_ground)
    climate_P_list.append(climate_P)
    climate_R_list.append(climate_R)
    
plt.plot(climate_R_list, climate_P_list)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for Climate')
plt.show()

# ngc
ngc_P_list , ngc_R_list = [], []
for i in range(5, 55, 5):
    ngc_prediction, ngc_diffs = Shot_Change_Detection(Load_Frames('ngc_out'), i / 1000, 0.1)
    ngc_prediction = [x + 1 for x in ngc_prediction]
    ngc_P , ngc_R = Precision_Recall(ngc_prediction, ngc_ground)
    ngc_P_list.append(ngc_P)
    ngc_R_list.append(ngc_R)
    
plt.plot(ngc_R_list, ngc_P_list)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve for NGC')
plt.show()




# 繪製以x軸為幀數、y軸為diffs的點圖
'''plt.figure(figsize=(10, 5))
plt.scatter(list(range(1, 1781)), climate_diffs, color='b', alpha=0.7)
plt.xlabel('Frame Number')
plt.ylabel('Histogram Difference')
plt.title('Histogram Difference for Climate Video')
plt.grid(True)
plt.tight_layout()
plt.show()'''

