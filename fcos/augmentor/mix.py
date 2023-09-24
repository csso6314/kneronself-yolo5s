import cv2
import numpy as np
import random


def mosaic(imgs, annos=None, inp_size=512):
    min_offset = 0.25
    inp_sizex = inp_sizey = inp_size

#     new_img = np.zeros((inp_sizey, inp_sizex, 3), 'float')
    new_anno = {'bboxes': np.zeros((0, 4)),
                'labels': np.zeros((0,))}
    # if random.getrandbits():
    easy_mode = random.getrandbits(1)
    for i in range(4):
        h,w,c = imgs[i].shape
        scale1 = 1.0*inp_sizex/w
        scale2 = 1.0*inp_sizey/h
        if easy_mode:
            scale = min(scale1, scale2)
        else:
            scale = np.random.uniform(min(scale1, scale2), max(scale1, scale2))

        h, w = int(h*scale), int(w*scale)
        imgs[i] = cv2.resize(imgs[i], (w,h),interpolation=np.random.randint(2))
        annos[i]['bboxes'] = annos[i]['bboxes']*scale

    if True:
        if easy_mode:
            xc = inp_sizex
            yc = inp_sizey
        else:
            xc = np.random.randint(inp_sizex * min_offset * 2, inp_sizex * (1 - min_offset) * 2)
            yc = np.random.randint(inp_sizey * min_offset * 2, inp_sizey * (1 - min_offset) * 2)
        for i in range(4):
            h, w, c = imgs[i].shape
            if i == 0:  # top left
                new_img = np.full((inp_sizey * 2, inp_sizex * 2, 3), 0, dtype='float')  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, inp_sizex * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(inp_sizey * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, inp_sizex * 2), min(inp_sizey * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            #             print i, x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b,new_img.shape
            new_img[y1a:y2a, x1a:x2a] = imgs[i][y1b:y2b, x1b:x2b]

            # new_img[:cut_y, :cut_x,:] = imgs[0][:cut_y, :cut_x,:]
            # new_img[:cut_y, cut_x:,:] = imgs[1][:cut_y, cut_x:,:]
            # new_img[cut_y:, :cut_x,:] = imgs[2][cut_y:, :cut_x,:]
            # new_img[cut_y:, cut_x:,:] = imgs[3][cut_y:, cut_x:,:]
            if annos is None or len(annos[i]['bboxes']) == 0:
                continue
            assert len(imgs) == len(annos)
            bboxes_tmp = annos[i]['bboxes'].copy()
            class_ids_tmp = annos[i]['labels'].copy()
            padw = x1a - x1b
            padh = y1a - y1b
            bboxes_tmp[:, [0, 2]] = bboxes_tmp[:, [0, 2]] + padw
            bboxes_tmp[:, [1, 3]] = bboxes_tmp[:, [1, 3]] + padh

            bboxes_tmp[:, [0, 2]] = np.clip(bboxes_tmp[:, [0, 2]], 0, inp_sizex * 2)
            bboxes_tmp[:, [1, 3]] = np.clip(bboxes_tmp[:, [1, 3]], 0, inp_sizey * 2)
            keep = np.logical_and(bboxes_tmp[:, 2] - bboxes_tmp[:, 0] > 2,
                                  bboxes_tmp[:, 3] - bboxes_tmp[:, 1] > 2)
            new_anno['bboxes'] = np.r_[new_anno['bboxes'], bboxes_tmp[keep]]
            new_anno['labels'] = np.r_[new_anno['labels'], class_ids_tmp[keep]]
        new_img = cv2.resize(new_img, (inp_size, inp_size), interpolation=np.random.randint(2))
        new_anno['bboxes'] = new_anno['bboxes'] / 2.
    # else:
    #     new_img = np.zeros((inp_sizey, inp_sizex, 3), 'float')
    #     cut_x = inp_sizex // 2
    #     cut_y = inp_sizex // 2
    #     for i in range(4):
    #         left = (i % 2) * cut_x
    #         right = inp_sizex if left else cut_x
    #         top = (i // 2) * cut_y
    #         bottom = inp_sizey if top else cut_y
    #         new_img[top:bottom, left:right, :] = cv2.resize(imgs[i], (cut_y, cut_x))
    #
    #         if annos is not None:
    #             assert len(imgs) == len(annos)
    #             bboxes_tmp = annos[i]['bboxes'].copy()
    #             class_ids_tmp = annos[i]['labels'].copy()
    #             bboxes_tmp = 1.0 * bboxes_tmp / 2
    #             bboxes_tmp[:, [0, 2]] = bboxes_tmp[:, [0, 2]] + left
    #             bboxes_tmp[:, [1, 3]] = bboxes_tmp[:, [1, 3]] + top
    #             keep = np.logical_and(bboxes_tmp[:, 2] - bboxes_tmp[:, 0] > 2,
    #                                   bboxes_tmp[:, 3] - bboxes_tmp[:, 1] > 2)
    #
    #             new_anno['bboxes'] = np.r_[new_anno['bboxes'], bboxes_tmp[keep]]
    #             new_anno['labels'] = np.r_[new_anno['labels'], class_ids_tmp[keep]]
    return new_img, new_anno
