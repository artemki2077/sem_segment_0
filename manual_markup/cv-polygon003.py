import cv2
import numpy as np
import os

colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255),
          5: (0, 255, 255), 6: (128, 128, 255), 7: (120, 255, 50), 8: (50, 160, 32), 9: (80, 5, 190), }



def on_mouse(event, x, y, flags, param):
    global poly, current_polygon, selected_class
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        poly[selected_class][current_polygon].append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN and len(poly[selected_class][current_polygon]) > 0:
        poly[selected_class][current_polygon].pop()
    for cur_class in range(len(poly)):
        for cur_polygon in range(len(poly[cur_class])):
            if len(poly[cur_class][cur_polygon]) > 0:
                cv2.circle(img2, poly[cur_class][cur_polygon][-1], 1, (0, 0, 255), -1)
            if len(poly[cur_class][cur_polygon]) > 1:
                for i in range(len(poly[cur_class][cur_polygon]) - 1):
                    cv2.circle(img2, poly[cur_class][cur_polygon][i], 2, (0, 0, 255), -1)
                    cv2.line(img=img2, pt1=poly[cur_class][cur_polygon][i], pt2=poly[cur_class][cur_polygon][i + 1],
                             color=colors[cur_class], thickness=1)
                cv2.circle(img2, poly[cur_class][cur_polygon][-1], 2, (0, 0, 255), -1)
                cv2.line(img=img2, pt1=poly[cur_class][cur_polygon][0], pt2=poly[cur_class][cur_polygon][-1],
                         color=colors[cur_class], thickness=1)
                mask = np.zeros(img2.shape, np.uint8)
                points = np.array(poly[cur_class][cur_polygon], np.int32)
                points = points.reshape((-1, 1, 2))
                mask = cv2.fillPoly(mask.copy(), [points], colors[cur_class])  # #
                img2 = cv2.addWeighted(src1=img2, alpha=1, src2=mask, beta=0.2, gamma=0)

    cv2.imshow('image', img2)


def create_blank_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blank_image = np.zeros_like(image)
    return blank_image


def create_mask_img(color_img):
    blank_img = create_blank_img(color_img)
    mask = np.zeros(blank_img.shape, np.uint8)
    for cur_class in range(len(poly)):
        for cur_polygon in range(len(poly[cur_class])):
            if len(poly[cur_class][cur_polygon]) > 2:
                points = np.array(poly[cur_class][cur_polygon], np.int32)
                points = points.reshape((-1, 1, 2))
                mask = cv2.fillPoly(mask.copy(), [points], (cur_class * 20 + 20,))
    return mask


print("[HELP] NUM key - choose class from 0 to 9")
print("[HELP] Left mouse click - add point to polygon")
print("[HELP] Right mouse click - remove point from polygon")
print("[HELP] Press X to add next polygon the same class")
print("[HELP] Press Z to return to the previous polygon the same class")
print("[HELP] Press space to save image mask")
print("[!!!!] Class with higher number puts mask on class with lower number")


work_folder = 'img_data/work/'
img_folder = 'img_data/images/'
mask_folder = 'img_data/masks/'


for i, filename in enumerate(list(os.walk(work_folder))[0][2]):
    print(f"Файл {i+1} из {len(list(os.walk(work_folder))[0][2])}", work_folder+filename)
    selected_class = 0
    current_polygon = 0
    poly = [[[]] for i in range(10)]
    img = cv2.imread(work_folder+filename)
    img = cv2.resize(img, (512, 512))
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    while True:
        key = cv2.waitKey(5)
        # if key != -1:
        #     print(key)
        if key == 120 or key == 247:
            if current_polygon == len(poly[selected_class]) - 1:
                poly[selected_class].append([])
            current_polygon += 1
            print("Class =", selected_class, 'Polygon number:', current_polygon)
        if key == 122 or key == 255 and current_polygon > 0:
            current_polygon -= 1
        if key == 32:
            mask_img = create_mask_img(img)
            cv2.imwrite(mask_folder+filename[:-3]+"png", mask_img)
            cv2.imwrite(img_folder+filename[:-3]+"png", img)
            os.remove(work_folder + filename)
            print("Mask has been saved")
            break
        if 48 <= key <= 57:
            selected_class = key - 48
            current_polygon = 0
            print(f"Class {selected_class} has been selected")
        if key == 27:
            break
    cv2.destroyAllWindows()
