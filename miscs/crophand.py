import os
import shutil
import sys
import json
import math
import cv2

CROP_SIZE = 160

filelist = [f for f in os.listdir(sys.argv[1]) if f.endswith('.jpg')]
savepath = os.path.join(sys.argv[1], 'cropped')
if not os.path.isdir(savepath):
    os.makedirs(savepath)
else:
    shutil.rmtree(savepath)
    os.makedirs(savepath)

for imgfile in filelist:
    img = cv2.imread(os.path.join(sys.argv[1], imgfile), cv2.IMREAD_GRAYSCALE)
    annofile = imgfile[:-3]+'json'
    with open(os.path.join(sys.argv[1], annofile), 'r') as f:
        anno = json.load(f)

        xsorted = sorted(anno['hand_pts'], cmp=lambda x,y: cmp(x[0], y[0]))
        ysorted = sorted(anno['hand_pts'], cmp=lambda x,y: cmp(x[1], y[1]))
        
        hand_pts_x_min = xsorted[0][0]
        idx = 0
        #while hand_pts_x_min <= 0 and xsorted[idx][2] == 0:
        while xsorted[idx][2] == 0:
            idx += 1
            hand_pts_x_min = xsorted[idx][0]
        hand_pts_x_max = xsorted[-1][0]
        hand_pts_y_min = ysorted[0][1]
        
        idx = 0
        #while hand_pts_y_min <= 0 and ysorted[idx][2] == 0:
        while ysorted[idx][2] == 0:
            idx += 1
            hand_pts_y_min = ysorted[idx][1]
        hand_pts_y_max = ysorted[-1][1]

        if 'hand_box_center' not in anno:
            anno['hand_box_center'] = []
            anno['hand_box_center'].append((hand_pts_x_max + hand_pts_x_min)/2)
            anno['hand_box_center'].append((hand_pts_y_max + hand_pts_y_min)/2)
        hand_x_center = anno['hand_box_center'][0]
        hand_y_center = anno['hand_box_center'][1]
        print('hand_center: %r' % (anno['hand_box_center']))

        print(hand_pts_x_min, hand_pts_x_max, hand_pts_y_min, hand_pts_y_max)
        if hand_pts_x_min < 0 or hand_pts_x_max >= img.shape[1] or hand_pts_y_min < 0 or hand_pts_y_max >= img.shape[0]:
            print("%s(%dx%d) anno(x=%f~%f, y=%f~%f) out of range, skip" % (imgfile, img.shape[1], img.shape[0], hand_pts_x_min, hand_pts_x_max, hand_pts_y_min, hand_pts_y_max))
            continue
        assert(hand_pts_x_min > 0)
        assert(hand_pts_x_max < img.shape[1])
        assert(hand_pts_y_min > 0)
        assert(hand_pts_y_max < img.shape[0])
        #print(crop_x_min, crop_x_max, crop_y_min, crop_y_max)

        crop_x_min = max(0, int(round(hand_x_center) - CROP_SIZE//2))
        if hand_pts_x_min < crop_x_min:
            crop_x_min = int(math.floor(hand_pts_x_min))
        
        crop_x_max = min(img.shape[1], crop_x_min + CROP_SIZE)
        if hand_pts_x_max > crop_x_max:
            crop_x_max = int(math.ceil(hand_pts_x_max))
        if crop_x_max == img.shape[1]:
            crop_x_min = max(0, crop_x_max - CROP_SIZE)

        crop_y_min = max(0, int(round(hand_y_center) - CROP_SIZE//2))
        if hand_pts_y_min < crop_y_min:
            crop_y_min = int(math.floor(hand_pts_y_min))
        
        crop_y_max = min(img.shape[0], crop_y_min + CROP_SIZE)
        if hand_pts_y_max > crop_y_max:
            crop_y_max = int(math.ceil(hand_pts_y_max))
        if crop_y_max == img.shape[0]:
            crop_y_min = max(0, crop_y_max - CROP_SIZE)
        
        print(img.shape, crop_x_min, crop_x_max, crop_y_min, crop_y_max)
        assert(crop_x_min >= 0)
        assert(crop_x_max <= img.shape[1])
        assert(crop_y_min >= 0)
        assert(crop_y_max <= img.shape[0])

        skip = False
        for hand_pt in anno['hand_pts']:
            hand_pt[0] -= crop_x_min
            hand_pt[1] -= crop_y_min
            #print(hand_pt[0], hand_pt[1], hand_pt[0] * CROP_SIZE/(crop_x_max - crop_x_min), hand_pt[1] * CROP_SIZE/(crop_y_max - crop_y_min))
            #if resize:
            #    hand_pt[0] = hand_pt[0] * CROP_SIZE / (crop_x_max - crop_x_min)
            #    hand_pt[1] = hand_pt[1] * CROP_SIZE / (crop_y_max - crop_y_min)
            if not hand_pt[0] < CROP_SIZE:
                print("%s anno x(%f) out of range, skip" % (imgfile, hand_pt[0]))
                skip = True
                break
            if not hand_pt[1] < CROP_SIZE:
                print("%s anno y(%f) out of range, skip" % (imgfile, hand_pt[1]))
                skip = True
                break
        
        if skip:
            continue
        
        anno['hand_box_center'][0] -= crop_x_min
        anno['hand_box_center'][1] -= crop_y_min
        anno['scale_provided'] = max((hand_pts_y_max-hand_pts_y_min)*1.2, (hand_pts_x_max-hand_pts_x_min)*1.2)/200.0
        
        with open(os.path.join(savepath, annofile), 'w') as fo:
            json.dump(anno, fo)
        
        print(img.shape)
        print(crop_x_min, crop_x_max, crop_y_min, crop_y_max)
        img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        print("Cropped %s" % (str(img.shape)))
        cv2.imwrite(os.path.join(savepath, imgfile[:-3]+'png'), img)