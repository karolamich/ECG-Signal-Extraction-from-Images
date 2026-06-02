import cv2
import numpy as np

img_test = cv2.imread('../img/P3_1.JPG')
# img_test = cv2.resize(img_test, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
# cv2.imshow('Image test', img_test) 

def get_chart_roi_v1(img):
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image1', img)
    _, img_thr = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_thr', img_thr)

    kernel = np.ones((6, 6), np.uint8)
    morph = cv2.morphologyEx(img_thr, cv2.MORPH_ERODE, kernel)
    # cv2.imshow('MORPH_ERODE', morph)

    kernel1 = np.ones((3, 3), np.uint8)
    morph1 = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel1)
    # cv2.imshow('MORPH_DILATE', morph1)

    kernel2 = np.ones((7, 7), np.uint8)
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_CLOSE, kernel2)
    # cv2.imshow('MORPH_CLOSE', morph2)

    contours, _ = cv2.findContours(morph2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_with_contours = img_copy.copy()
    # cv2.drawContours(image=image_with_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4)
    # cv2.imshow('image_with_contours', image_with_contours)
    
    sorted_list = sorted(contours, key=cv2.contourArea)
    largest_contour = sorted_list[-2]

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img_copy[y:y+h, x:x+w]
    cv2.imshow('cropped_img', cropped_img)

    return cropped_img, (x, y, w, h)

def get_chart_roi(img):
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image1', img)
    _, img_thr = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow('img_thr', img_thr)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(img_thr, cv2.MORPH_DILATE, kernel)
    # cv2.imshow('MORPH_DILATE', morph)

    kernel1 = np.ones((6, 6), np.uint8)
    morph1 = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel1)
    # cv2.imshow('MORPH_ERODE', morph1)

    kernel = np.ones((3, 3), np.uint8)
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_DILATE, kernel)
    # cv2.imshow('MORPH_DILATE2', morph2)

    kernel1 = np.ones((6, 6), np.uint8)
    morph3 = cv2.morphologyEx(morph2, cv2.MORPH_ERODE, kernel1)
    # cv2.imshow('MORPH_ERODE2', morph3)

    kernel2 = np.ones((6, 6), np.uint8)
    morph4 = cv2.morphologyEx(morph3, cv2.MORPH_CLOSE, kernel2)
    # cv2.imshow('MORPH_CLOSE', morph4)

    contours, _ = cv2.findContours(morph2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    image_with_contours = img_copy.copy()
    cv2.drawContours(image=image_with_contours, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4)
    # cv2.imshow('image_with_contours', image_with_contours)
    
    sorted_list = sorted(contours, key=cv2.contourArea)
    largest_contour = sorted_list[-2]

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img_copy[y:y+h, x:x+w]
    # cv2.imshow('cropped_img', cropped_img)

    return cropped_img, (x, y, w, h)

def get_reference_lines(img):
    tolerance_angle = 0
    bound_limit = 20
    height, width = img.shape[:2]

    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image ref gray', img_gray)

    _, img_thr = cv2.threshold(img_gray, 40, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image gray threshold', img_thr)
    
    img_processed = img_thr

    lines = cv2.HoughLinesP(
                img_processed, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=50, # Min allowed length of line
                maxLineGap=30 # Max allowed gap between line for joining them
                )

    horizontal_lines = []

    for points in lines:
        x1,y1,x2,y2=points[0]
        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        abs_angle = np.abs(angle_deg)

        if y1 <= bound_limit or y2 <= bound_limit:
            # cv2.line(img_copy,(x1,y1),(x2,y2),(255, 0, 0),1)
            # print('Usunięto linie blisko góry zdjęcia')
            pass

        elif y1 >= height-bound_limit or y2 >= height-bound_limit:
            # cv2.line(img_copy,(x1,y1),(x2,y2),(255, 0, 0),1)
            # print('Usunięto linie blisko dołu zdjęcia')
            pass

        elif abs_angle <= tolerance_angle or abs(180-abs_angle) <= tolerance_angle :
            cv2.line(img_copy,(x1,y1),(x2,y2),(0, 255, 0),1)
            horizontal_lines.append((y1, (x1, y1, x2, y2)))
    
    # cv2.imshow('Image with lines', img_copy)
    
    lines_array = np.array([line[0] for line in horizontal_lines])
    lines_array.sort()
    # print('Ys:', lines_array)

    # bins_distance = int(np.floor(height/13))
    ref_val = lines_array[0]
    val_list = []
    ref_lines = []

    # print(f'Len lines array: {len(lines_array)}')
    # print(range(1, len(lines_array)-1, 1))
    for i in range(1, len(lines_array), 1):

        # print(lines_array[i], i, len(lines_array))

        if i == len(lines_array)-1:
            # print('Test')
            mean = int(np.floor(np.mean(val_list)))
            ref_lines.append(mean)
        elif abs(lines_array[i]-ref_val) < bound_limit:
            val_list.append(lines_array[i])
            ref_val = lines_array[i]

        else:
            # print(val_list)
            mean = int(np.floor(np.mean(val_list)))
            ref_lines.append(mean)
            val_list = []
            ref_val = lines_array[i]
            val_list.append(lines_array[i])
        
    # print('Reference lines:', ref_lines)
    # print('Ys:', lines_array)
    if len(ref_lines) != 12:
        print(f'Wrong number od lines. Expected: 12, found: {len(ref_lines)}')

    ref_lines_copy = img.copy()

    for line in ref_lines:
        cv2.line(ref_lines_copy, (0, line), (width, line), (0, 0, 255), 1)
    
    # cv2.imshow('Image with lines', ref_lines_copy)
    return ref_lines



# get_chart_roi(img_test)

img_cropped, _ = get_chart_roi(img_test)

get_reference_lines(img_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()