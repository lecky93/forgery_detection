# import numpy as np
# from sklearn.metrics import roc_auc_score
#
# y = np.array([0, 0, 1, 1])  # 真实值
# y_pred1 = np.array([0.3, 0.2, 0.25, 0.7])  # 预测值
# y_pred2 = np.array([0, 0, 1, 0])  # 预测值
#
# auc_score1 = roc_auc_score(y, y_pred1)
# print(auc_score1)  # 0.75
#
# auc_score2 = roc_auc_score(y, y_pred2)
# print(auc_score2)  # 0.75


import os
import cv2
import numpy as np


folder = r'D:\workspace\python\dataset\forgery\nist16\ann'
path = os.listdir(folder)
for item in path:
    img = cv2.imread(os.path.join(folder, item), 1)
    img_np = np.array(img)
    if img_np.sum() == 0:
        print(item)