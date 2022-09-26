import cv2
import os
import numpy as np

def Resize(scale, files, in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for file in files:
        img = cv2.imread(os.path.join(in_folder, file))
        dim = img.shape[:2]
        dim = (int(dim[0] * scale), int(dim[1] * scale))
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_folder, file), img)

def GaussianBlur(kenel_size, files, in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for file in files:
        img = cv2.imread(os.path.join(in_folder, file))
        img = cv2.GaussianBlur(img, kenel_size, 0)
        cv2.imwrite(os.path.join(out_folder, file), img)

def GaussianNoise(sigma, files, in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for file in files:
        img = cv2.imread(os.path.join(in_folder, file))
        gauss = np.random.normal(0, sigma, img.shape)
        noisy_img = img + gauss
        noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
        cv2.imwrite(os.path.join(out_folder, file), noisy_img)

def JpegCompress(quality, files, in_folder, out_folder):
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    for file in files:
        img = cv2.imread(os.path.join(in_folder, file))
        cv2.imwrite(os.path.join(out_folder, file), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

if __name__ == '__main__':
    ori_path = r'D:\workspace\python\dataset\forgery\casia\CASIA2\image'
    files = os.listdir(ori_path)

    # resize_78 = r'D:\workspace\python\dataset\forgery\nist16\image_resize_78'
    # resize_25 = r'D:\workspace\python\dataset\forgery\nist16\image_resize_25'
    #
    # gaussianBlur_3 = r'D:\workspace\python\dataset\forgery\nist16\image_gaussianBlur_3'
    # gaussianBlur_5 = r'D:\workspace\python\dataset\forgery\nist16\image_gaussianBlur_5'
    #
    # gaussianNoise_3 = r'D:\workspace\python\dataset\forgery\nist16\image_gaussianNoise_3'
    # gaussianNoise_15 = r'D:\workspace\python\dataset\forgery\nist16\image_gaussianNoise_15'
    #
    # jpeg_100 = r'D:\workspace\python\dataset\forgery\nist16\image_jpeg_100'
    # jpeg_50 = r'D:\workspace\python\dataset\forgery\nist16\image_jpeg_50'

    # Resize(0.78, files, ori_path, resize_78)
    # Resize(0.78, files, ori_path, resize_25)
    #
    # GaussianBlur((3,3), files, ori_path, gaussianBlur_3)
    # GaussianBlur((5,5), files, ori_path, gaussianBlur_5)
    #
    # GaussianNoise(3, files, ori_path, gaussianNoise_3)
    # GaussianNoise(15, files, ori_path, gaussianNoise_15)
    #
    # JpegCompress(100, files, ori_path, jpeg_100)
    # JpegCompress(50, files, ori_path, jpeg_50)

    gaussianBlur = r'D:\workspace\python\dataset\forgery\casia\CASIA2\image_gaussianBlur_'
    jpeg = r'D:\workspace\python\dataset\forgery\casia\CASIA2\image_jpeg_'


    for i in range(90, 40, -10):
        path = jpeg + str(i)
        JpegCompress(i, files, ori_path, path)

    for i in range(5, 30, 6):
        path = gaussianBlur + str(i)
        GaussianBlur((i, i), files, ori_path, path)