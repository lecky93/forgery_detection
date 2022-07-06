import os
import cv2

def suffix2jpg(ori_path, target_path):
    path = ori_path
    path2 = target_path

    if not os.path.exists(path2):
        os.mkdir(path2)

    images = os.listdir(path)
    for item in images:
        image_prefix = item.split('.')[0]
        suffix = item.split('.')[1]
        if suffix != 'jpg' and suffix != 'tif':
            continue

        image_path = os.path.join(path, item)
        print(image_path)

        image = cv2.imread(image_path)
        new_path = image_prefix + '.jpg'
        cv2.imwrite(os.path.join(path2, new_path), image)


def mask2ann(mask_path, ann_path, convert=False, add_gt=False):
    path = mask_path
    path2 = ann_path
    if not os.path.exists(path2):
        os.mkdir(path2)
    images = os.listdir(path)
    for item in images:
        image_prefix = item.split('.')[0]
        # suffix = item.split('.')[1]
        # if suffix != 'png':
        #     continue
        image_path = os.path.join(path, item)
        print(image_path)
        image = cv2.imread(image_path, 0)
        new_path = image_prefix + '.png'
        if add_gt:
            new_path = image_prefix + '_gt.png'

        image = image // 128
        if convert:
            image[image == 1] = 2
            image[image == 0] = 1
            image[image == 2] = 0


        cv2.imwrite(os.path.join(path2, new_path), image)

def ann2mask(ann_path, mask_path):
    img_root = ann_path
    imgs = os.listdir(img_root)
    new_path = mask_path
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for item in imgs:
        img = cv2.imread(os.path.join(img_root, item), 0)
        img[img == 1] = 255
        cv2.imwrite(os.path.join(new_path, item), img)

def generate_splittxt():
    path = r'D:\dataset\forgery\casia\CASIA1\image'
    imgs_path = os.listdir(path)

    f1 = open('D:/dataset/forgery/casia/CASIA1/copy-move.txt', 'w')
    f2 = open('D:/dataset/forgery/casia/CASIA1/splicing.txt', 'w')

    for item in imgs_path:
        name = os.path.splitext(item)[0]
        if item.split('_')[1] == 'S':
            f1.write(name + '\n')

        else:
            f2.write(name + '\n')

if __name__ == '__main__':
    mask2ann(r'D:\dataset\forgery\nist16\mask', r'D:\dataset\forgery\nist16\ann', convert=True, add_gt=True)
    # ann2mask(r'D:\dataset\forgery\nist16\ann', r'D:\dataset\forgery\nist16\test')