import cv2
import numpy as np
import os

def create_blank_img(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blank_image = np.zeros_like(image)
    return blank_image


def region_of_interest (image, vertices):
    msk = np.zeros_like(image)
    ch_count = image.shape[2]
    match_mask_color = (255,) * ch_count
    cv2.fillPoly(msk, vertices, match_mask_color)
    masked_img = cv2.bitwise_and(image, msk)
    return masked_img


def add_white_lane(image, blank_image):
    h_min = np.array((47, 4, 190), np.uint8)
    h_max = np.array((101, 57, 255), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    region_vertices = [(0, height), (0, height // 2), (width, height // 2), (width, height)]
    hsv = region_of_interest(hsv, np.array([region_vertices], np.int32), )
    thresh = cv2.inRange(hsv, h_min, h_max)
    gray_mask = np.zeros_like(thresh)
    gray_mask[:]=100
    gray_mask = cv2.bitwise_and(thresh,gray_mask)
    blank_image = cv2.bitwise_or(blank_image, gray_mask)
    return blank_image


def add_yellow_lane(image, blank_image):
    h_min = np.array((17, 60, 190), np.uint8)
    h_max = np.array((43, 228, 255), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    region_vertices = [(0, height), (0, height // 2), (width, height // 2), (width, height)]
    hsv = region_of_interest(hsv, np.array([region_vertices], np.int32), )
    thresh = cv2.inRange(hsv, h_min, h_max,)
    gray_mask = np.zeros_like(thresh)
    gray_mask[:] = 150
    gray_mask = cv2.bitwise_and(thresh, gray_mask)
    blank_image = cv2.bitwise_or(blank_image, gray_mask)
    return blank_image


def add_red_lane(image, blank_image):
    h_min = np.array((0, 76, 118), np.uint8)
    h_max = np.array((26, 167, 255), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    region_vertices = [(0, height), (0, height // 2), (width, height // 2), (width, height)]
    hsv = region_of_interest(hsv, np.array([region_vertices], np.int32), )
    thresh = cv2.inRange(hsv, h_min, h_max, )
    gray_mask = np.zeros_like(thresh)
    gray_mask[:] = 200
    gray_mask = cv2.bitwise_and(thresh, gray_mask)
    blank_image = cv2.bitwise_or(blank_image, gray_mask)
    return blank_image


def add_road(image, blank_image):
    h_min = np.array((0, 0, 7), np.uint8)
    h_max = np.array((56, 92, 115), np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    width = hsv.shape[1]
    region_vertices = [(0, height), (0, height // 3), (width, height // 3), (width, height)]
    hsv = region_of_interest(hsv, np.array([region_vertices], np.int32), )
    thresh = cv2.inRange(hsv, h_min, h_max, )
    gray_mask = np.zeros_like(thresh)
    gray_mask[:] = 50
    gray_mask = cv2.bitwise_and(thresh, gray_mask)
    blank_image = cv2.bitwise_or(blank_image, gray_mask)
    return blank_image


def segment_image(image):
    mask = create_blank_img(image)
    mask = add_road(img, mask)
    mask = add_white_lane(img, mask)
    mask = add_yellow_lane(img, mask)
    mask = add_red_lane(img, mask)
    return mask


img_folder = 'image/'
mask_folder = 'mask/'

for filename in list(os.walk(img_folder))[0][2]:
    img = cv2.imread(img_folder+filename)
    segmented_image = segment_image(img)
    cv2.imwrite(mask_folder+filename[:-3]+"png", segmented_image)
    print(filename)
