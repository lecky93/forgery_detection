from skimage import io, measure, data
import cv2, os

if __name__ == '__main__':
    path = r'D:\workspace\python\dataset\forgery\casia\CASIA2'

    image_path = os.path.join(path, 'image')
    mask_path = os.path.join(path, 'mask')

    cut_image_path = os.path.join(path, 'image_cut')
    cut_mask_path = os.path.join(path, 'mask_cut')

    if not os.path.exists(cut_image_path):
        os.mkdir(cut_image_path)
    if not os.path.exists(cut_mask_path):
        os.mkdir(cut_mask_path)

    image_files = os.listdir(image_path)

    for filename in image_files:
        name = filename.split('.')[0]
        maskname = name+'_gt.png'

        img = cv2.imread(os.path.join(image_path, filename))
        mask = cv2.imread(os.path.join(mask_path, maskname), 0)

        mask[mask < 128] = 0
        mask[mask >= 128] = 255

        H, W, C = img.shape

        area = mask.sum() / 255
        ratio = area / (W * H)

        if ratio > 0.3:
            cv2.imwrite(os.path.join(image_path, filename), img)
            cv2.imwrite(os.path.join(mask_path, maskname), mask)


        labeld_mask, num = measure.label(mask, background=0, return_num=True)

        index = 0
        for region in measure.regionprops(labeld_mask):
            area = region.area
            area_bbox = region.area_bbox

            if area < 50:
                continue

            minr, minc, maxr, maxc = region.bbox

            bbox_w = maxc - minc
            bbox_h = maxr - minr

            padding_w = int(bbox_w / 5)
            padding_h = int(bbox_h / 5)

            min_x = minc - padding_w
            max_x = maxc + padding_w
            min_y = minr - padding_h
            max_y = maxr + padding_h

            if min_x < 0:
                min_x = 0
            if max_x >= W:
                max_x = W - 1

            if min_y < 0:
                min_y = 0
            if max_y >= H:
                max_Y = H - 1

            img_cut = img[min_y:max_y, min_x:max_x]
            mask_cut = mask[min_y:max_y, min_x:max_x]

            img_cut_name = os.path.join(cut_image_path, name + '_' + str(index) + '.jpg')
            mask_cut_name = os.path.join(cut_mask_path, name + '_' + str(index) + '_gt.png')

            cv2.imwrite(img_cut_name, img_cut)
            cv2.imwrite(mask_cut_name, mask_cut)
            index += 1
