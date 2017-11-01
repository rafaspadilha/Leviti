# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt

import numpy as np
import cv2
import scipy

import os
import sys
import shutil

# define default values from database
IMG_SIZE = 224
MIN_MATCH_COUNT = 1

ignore = True

OUTPUT = 'aligned'
root   = '/work/rpadilha/AMOS/'

def cv2keras(img):
    return np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)

def register(original, new, transform = True):
    try:
        img = cv2.imread(new)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(original, None)

        if img is None:
            return None
        else:
            kp2, des2 = sift.detectAndCompute(img, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        # sanity test ftw!
        if des1 is not None and des2 is not None and \
           len(des1) >= 2 and len(des2) >= 2:
            matches = flann.knnMatch(np.asarray(des1, np.float32), \
                                     np.asarray(des2, np.float32), k=2)
        else:
            return None

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        # is it useful for us?
        if len(good) < MIN_MATCH_COUNT:
            print "Images are not sufficiently alike - %d/%d" % (len(good), MIN_MATCH_COUNT)
            matchesMask = None

            return None

        if len(good) > MIN_MATCH_COUNT and len(good) < MIN_MATCH_COUNT * 2:
            return img

        if transform and len(good) > MIN_MATCH_COUNT * 2:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if mask is not None:
                matchesMask = mask.ravel().tolist()

            h, w, _ = original.shape

            if M is not None and M.shape == (3, 3):
                res = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            else:
                res = img
        else:
            res = img

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #                    singlePointColor = None,
        #                    matchesMask = matchesMask, # draw only inliers
        #                    flags = 2)

        # img3 = cv2.drawMatches(original, kp1, img, kp2, good, None, **draw_params)

        return res

    except IOError as e:
        print ('Could not read: ', img_f, ', skipping it!')

        return None

def load_data(folder, min_obj, save):
    sub_f = sorted(os.listdir(folder))

    print folder, "  " , sub_f

    total = 0
    model = None

    for s in sub_f:
        sub_n = folder + '/' + s
        img_f = os.listdir(sub_n)

        # initialize!
        if model is None:
            m     = sub_n + '/' + img_f[0] # first image, taken as a model
            model = cv2.imread(m)

            # get specs of image
            h = model.shape[0]
            w = model.shape[1]

        for img in img_f:
            file =  sub_n + '/' + img

            try:
                # get full image, following registration
                data = register(model, file, False)

                # if registration went bad, simply ignore it
                if data is None:
                    continue

                total  += 1
                num_img = 0 # counter

                # check specs of image
                if h != data.shape[0] or w != data.shape[1]:
                    continue

                # crop conners to image default size
                for i in range(min(h/IMG_SIZE + 1, 2)):
                    for j in range(min(w/IMG_SIZE + 1, 2)):
                        cropped = data[(h - IMG_SIZE) * i:(h - IMG_SIZE) * i + IMG_SIZE,
                                       (w - IMG_SIZE) * j:(w - IMG_SIZE) * j + IMG_SIZE, :]

                        if not os.path.exists(save + str(num_img)):
                            os.makedirs(save + str(num_img))

                        label = save + str(num_img) + '/' + img

                        avg_color = np.average(np.average(cropped, axis = 0), axis = 0)

                        # sanity check for the image quality
                        if ignore or not all (i >= 230 for i in avg_color):
                            cv2.imwrite(label, cropped)

                            num_img += 1
                        else:
                            del label
                            del cropped

                            print ('Incorrect image.')

                # now, crop the center and be done with it!
                if h/IMG_SIZE != 0 or w/IMG_SIZE != 0:
                    cropped = data[h/2 - IMG_SIZE/2:h/2 + IMG_SIZE/2, 
                                   w/2 - IMG_SIZE/2:w/2 + IMG_SIZE/2, :]

                    if not os.path.exists(save + str(num_img)):
                        os.makedirs(save + str(num_img))

                    label = save + str(num_img) + '/' + img

                    avg_color = np.average(np.average(cropped, axis = 0), axis = 0)

                    # sanity check for the image quality
                    if ignore or not all (i >= 230 for i in avg_color):
                        cv2.imwrite(label, cropped)

                        num_img += 1
                    else:
                        del label
                        del cropped
                            
                        print ('Incorrect image.')

                else:
                    # in case the image is too small for any cropping, simply resize it
                    cropped = scipy.misc.imresize(data, 
                                                  (IMG_SIZE, IMG_SIZE), 
                                                  interp='bilinear')

                    if not os.path.exists(save + str(num_img)):
                        os.makedirs(save + str(num_img))

                    print ('Image is too small.')

                    label = save + str(num_img) + '/' + img

                    cv2.imwrite(label, cropped)

                    num_img += 1

            except IOError as e:
                print ('Could not read: ', img_f, ', skipping it!')

    if total < min_obj:
        raise Exception('Fewer images than expected. Skipping folder ', folder, 
                        '!')

        shutil.rmtree(save)

        return False

    return True

if __name__ == "__main__":
    for d in os.listdir(root):
        # ignore final result
        if d == OUTPUT:
            continue

        try:
            dataset = os.path.join(root, d)
            new_dataset = os.path.join(root, os.path.join(OUTPUT, d))

            if not os.path.exists(new_dataset):
                os.makedirs(new_dataset)

            if not load_data(dataset, 10, new_dataset + '/'):
                shutil.rmtree(new_dataset)

        except IOError as e:
            print ('Could not create folder: ', dataset)
