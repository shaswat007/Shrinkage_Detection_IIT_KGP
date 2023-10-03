import csv
import os
import time
from itertools import count
from queue import Queue

import cv2
import numpy as np


def in_range(i, j, im_shape):
    return 0 <= i < im_shape[0] and 0 <= j < im_shape[1]


def bfs_black(img, par, i, j, par_idx):
    # print("call", i, j)
    deltas = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1)
    ]
    x_sum = y_sum = 0

    q = Queue(0)
    q.put((i, j))
    par[i, j] = par_idx
    c = 1

    while not q.empty():
        i1, j1 = q.get()
        x_sum += i1
        y_sum += j1
        for d in deltas:
            i2, j2 = i1 + d[0], j1 + d[1]
            # print("test", i1, j1)

            if in_range(i2, j2, img.shape) and img[i2, j2] == 0 and par[i2, j2] == 0:
                # print("i1j1", i1, j1)
                q.put((i2, j2))
                c += 1
                par[i2, j2] = par_idx

    return c, x_sum / c, y_sum / c


def bfs_white(img, par, i, j, par_idx, valid_blobs):
    # print("call", i, j)
    deltas = [
        (-1, -1), (-1, 0), (-1, 1), (0, 1),
        (1, 1), (1, 0), (1, -1), (0, -1)
    ]

    q = Queue(0)
    q.put((i, j))
    par[i, j] = par_idx
    nbds = set()

    while not q.empty():
        i1, j1 = q.get()
        for d in deltas:
            i2, j2 = i1 + d[0], j1 + d[1]
            # print("test", i1, j1)
            if not in_range(i2, j2, img.shape):
                continue

            if img[i2, j2] == 255 and par[i2, j2] == 0:
                # print("i1j1", i1, j1)
                q.put((i2, j2))
                par[i2, j2] = par_idx
            elif img[i2, j2] == 0:
                if par[i2, j2] in valid_blobs:
                    nbds.add(par[i2, j2])

    return len(nbds) == 1


def flood_fill(img):
    h, w = img.shape
    par = np.zeros_like(img, dtype=np.uint32)
    c = count()
    next(c)
    valid_blobs = set()
    to_whiten = set()

    # REVIEW: See if this threshold makes sense,
    #         currently little less than 1% of image area
    mass_threshold = 50000 * img.shape[0] * img.shape[1] / (2592 * 1944)

    for i in range(h):
        for j in range(w):
            if img[i, j] == 0 and par[i, j] == 0:
                # print('stage1', i, j, f'{100 * i / h:.2f}%')
                idx = next(c)
                res, _, _ = bfs_black(img, par, i, j, idx)
                # print('stage1', i, j, f'{100 * i / h:.2f}% -> {res}')
                if res > mass_threshold:
                    valid_blobs.add(idx)
                else:
                    to_whiten.add(idx)

    # print('---------------------------------')
    # print(c, len(valid_blobs), len(to_whiten))
    # print('---------------------------------')

    to_fill = set()
    for i in range(h):
        for j in range(w):
            if img[i, j] == 255 and par[i, j] == 0:
                # print('stage2', i, j, f'{100 * i / h:.2f}%')
                idx = next(c)
                res = bfs_white(img, par, i, j, idx, valid_blobs)
                # print('stage2', i, j, f'{100 * i / h:.2f}% -> {res}')
                if res:
                    to_fill.add(idx)

    print(c, len(to_fill))

    img_copy = img.copy()
    for i in range(h):
        for j in range(w):
            if img_copy[i, j] == 255 and par[i, j] in to_fill:
                img_copy[i, j] = 0
            elif img_copy[i, j] == 0 and par[i, j] in to_whiten:
                img_copy[i, j] = 255
    return img_copy


def compute_areas(img):
    h, w = img.shape
    par = np.zeros_like(img, dtype=np.uint32)
    c = count()
    next(c)

    areas = []

    for i in range(h):
        for j in range(w):
            if img[i, j] == 0 and par[i, j] == 0:
                # print('stage1', i, j, f'{100 * i / h:.2f}%')
                idx = next(c)
                res, x_avg, y_avg = bfs_black(img, par, i, j, idx)
                areas.append([res, x_avg, y_avg])
                # print(res, x_avg, y_avg)
                # print('stage1', i, j, f'{100 * i / h:.2f}% -> {res}')
                # if res:
                #     valid_blobs.add(idx)
                # else:
                #     to_whiten.add(idx)
    return areas


def main():
    bbt1 = time.time()
    # TODO: Create dirs out and areas if not exist
    # Get all image paths. Currently working with the first image to test
    img_paths = os.listdir('data')
    img_color = None
    skip = 4
    for idx1, img_path in enumerate(sorted(img_paths)):
        if idx1 % skip != 0:
            continue
        bt1 = time.time()
        print(idx1, img_path)
        idx = idx1 // skip
        # img_path = 'E:\\Py\\is_proj\\data\\2021-08-18 08_35_36.jpg'
        # img_path = 'E:\\Py\\is_proj\\data\\2021-08-04 13_11_45.jpg'
        # Read in the file in color mode
        img_color = cv2.imread(os.path.join('data/', img_path), cv2.IMREAD_COLOR)
        img_color = cv2.resize(img_color, (648, 486))

        # Display the img and save it
        # cv2.imshow("Original input image", cv2.resize(img_color, (648, 486)))
        # cv2.waitKey(0)
        # os.mkdir(f'out_imgs/{idx:03d}/')
        cv2.imwrite(f'out_imgs/{idx:03d}/01_input.png', img_color)

        # Convert to HSV color system to perform Color Thresholding
        # Hue values between 36 and 70 identified for green.
        # Saturation and Value (Brightness) kept
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(img_hsv, (20, 36, 42), (76, 255, 255))

        mask = color_mask > 0
        img_color_masked = np.ones_like(img_color, np.uint8) * 255
        img_color_masked[mask] = img_color[mask]
        # Display the img and save it
        # cv2.imshow("Color masked image", cv2.resize(img_color_masked, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/02_color_masked.png', img_color_masked)
        # return

        # Invert the mask to generate a black-and-white image
        img_bw = np.ones_like(color_mask, np.uint8) * 255
        img_bw[mask] = 0
        # cv2.imshow("Black and white image", cv2.resize(img_bw, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/03_masked_bw.png', img_bw)

        # Perform Floodfill algorithm, also measure time
        t1 = time.time()
        out = flood_fill(img_bw)
        t2 = time.time()
        print(f'{(t2 - t1) // 60} minutes and {(t2 - t1) % 60:.2f} seconds')

        # Display the images (including bw for comparison) and save
        # cv2.imshow("Original", cv2.resize(img_bw, (648, 486)))
        # cv2.waitKey(0)
        # cv2.imshow("After Floodfill", cv2.resize(out, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/04_floodfill.png', out)

        # Perform morphological image "Opening" operation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        res = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Image Opening 1", cv2.resize(res, (648, 486)))
        cv2.imwrite(f'out_imgs/{idx:03d}/05_open1.png', res)
        # cv2.waitKey(0)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Image Opening 2", cv2.resize(res, (648, 486)))
        cv2.imwrite(f'out_imgs/{idx:03d}/06_open2.png', res)
        # cv2.waitKey(0)

        t1 = time.time()
        out = flood_fill(res)
        # flood_fill(cv2.resize(bw, (648, 486)))
        t2 = time.time()
        print(f'{(t2 - t1) // 60} minutes and {(t2 - t1) % 60:.2f} seconds')

        # cv2.imshow("Original 2", cv2.resize(img_bw, (648, 486)))
        # cv2.waitKey(0)
        # cv2.imshow("After Floodfill 2", cv2.resize(out, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/07_flood2.png', out)

        res = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Image Opening 2 1", cv2.resize(res, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/08_open3.png', res)
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
        # cv2.imshow("Image Opening 2 2", cv2.resize(res, (648, 486)))
        # cv2.waitKey(0)
        cv2.imwrite(f'out_imgs/{idx:03d}/09_open4.png', res)

        areas = compute_areas(res)
        with open(f'areas/{idx:03d}.csv', 'w', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)

            # write a row to the csv file
            for row in areas:
                writer.writerow(row)

        # Run canny edge detection to isolate edges
        edged = cv2.Canny(res, 30, 200)
        cv2.imwrite(f'out_imgs/{idx:03d}/10_canny.png', edged)

        # Perform contour detection
        contours, hierarchy = cv2.findContours(255 - res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours1, hierarchy1 = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c_img = np.zeros_like(img_color)
        c_img1 = np.zeros_like(img_color)
        cv2.drawContours(c_img, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawContours(c_img1, contours1, -1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(f'out_imgs/{idx:03d}/11_contours_normal.png', c_img)
        cv2.imwrite(f'out_imgs/{idx:03d}/12_contours_canny.png', c_img1)

        bt2 = time.time()
        print(f'Loop for 1 img: {(bt2 - bt1) // 60} minutes and {(bt2 - bt1) % 60:.2f} seconds')
        # break

    bbt2 = time.time()
    print(f'TOTAL: {(bbt2 - bbt1) // 60} minutes and {(bbt2 - bbt1) % 60:.2f} seconds')


if __name__ == '__main__':
    main()
