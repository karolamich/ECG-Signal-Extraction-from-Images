import cv2
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os


# ------------PRZELICZANIE DLUGOSCI REFERENCYJNEJ OSI X-----------

def get_ref_scale_in_pixel(img):
    tolerance_angle = 5
    img_ref = img
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image ref gray', img_ref_gray)

    _, img_ref_thr = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image ref gray threshold', img_ref_thr)

    width = img_ref.shape[1]
    width_low = int(0.4*width)
    width_up = int(0.6*width)

    roi = img_ref_gray[:, width_low: width_up]
    # cv2.imshow('ROI', roi)
    edges = cv2.Canny(roi,50,150,apertureSize=3)
    vertical_lines =[]
    lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180,
                threshold=200, 
                minLineLength=600,
                maxLineGap=5 
                )
    x_list = []
    for points in lines:
        x1,y1,x2,y2=points[0]
        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        if abs(abs(angle_deg)-90) <= tolerance_angle:
            rand_col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.line(img_ref,(width_low+x1,y1),(width_low+x2,y2),rand_col,1)
            x_list.append(x1)
            vertical_lines.append([(x1,y1),(x2,y2)])

    x_list.sort()
    x_min = np.mean(x_list[:2])
    x_max = np.mean(x_list[-2:])
    ref_len_in_pixel = x_max - x_min

    return int(ref_len_in_pixel)

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
                img_processed,
                1, 
                np.pi/180, 
                threshold=50, 
                minLineLength=15, 
                maxLineGap=30 
                )

    horizontal_lines = []

    for points in lines:
        x1,y1,x2,y2=points[0]
        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        abs_angle = np.abs(angle_deg)

        if y1 <= bound_limit or y2 <= bound_limit:
            # print('Usunięto linie blisko góry zdjęcia')
            pass

        elif y1 >= height-bound_limit or y2 >= height-bound_limit:
            # print('Usunięto linie blisko dołu zdjęcia')
            pass

        elif abs_angle <= tolerance_angle or abs(180-abs_angle) <= tolerance_angle :
            cv2.line(img_copy,(x1,y1),(x2,y2),(0, 255, 0),1)
            horizontal_lines.append((y1, (x1, y1, x2, y2)))
    
    # cv2.imshow('Image with lines', img_copy)
    
    lines_array = np.array([line[0] for line in horizontal_lines])
    lines_array.sort()

    ref_val = lines_array[0]
    val_list = []
    ref_lines = []

    for i in range(1, len(lines_array), 1):
        if i == len(lines_array)-1:
            mean = int(np.floor(np.mean(val_list)))
            ref_lines.append(mean)
        elif abs(lines_array[i]-ref_val) < bound_limit:
            val_list.append(lines_array[i])
            ref_val = lines_array[i]
        else:
            mean = int(np.floor(np.mean(val_list)))
            ref_lines.append(mean)
            val_list = []
            ref_val = lines_array[i]
            val_list.append(lines_array[i])
        
    if len(ref_lines) != 12:
        print(f'Wrong number od lines. Expected: 12, found: {len(ref_lines)}')

    ref_lines_copy = img.copy()

    for line in ref_lines:
        cv2.line(ref_lines_copy, (0, line), (width, line), (0, 0, 255), 1)
    
    # cv2.imshow('Image with lines', ref_lines_copy)
    return ref_lines

def find_white_points(img, img_mod, ref_pt, w):
    i = 1
    while True:
        if ref_pt > 0 and ref_pt <= img.shape[0]-1:
            ref_pt = np.clip(ref_pt, 0, img.shape[0]-1)
            if  ref_pt+i <= img.shape[0]-1 and img[ref_pt+i, w] == 255 and ref_pt+i < img.shape[0]:
                img_mod[ref_pt+i, w] = (0, 0, 255)
                ref_pt += i
                break
            elif img[ref_pt, w] == 255:
                img_mod[ref_pt, w] = (0, 0, 255)
                break
            elif img[ref_pt-i, w] == 255:
                img_mod[ref_pt-i, w] = (0, 0, 255)
                ref_pt -= i
                break
            else:
                i += 1

    return ref_pt, img_mod

def find_points(img, ref_lines):
    points = {}
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image gray threshold', img_thr)

    img_copy = cv2.cvtColor(img_thr, cv2.COLOR_GRAY2BGR)
    idx = 1
    for y_ref in ref_lines:
        chart_data = []
        ref_pt = y_ref
        # cv2.line(img_copy, (0, y_ref), (img_thr.shape[1], y_ref), (0, 255, 0), 1)
        for w in range(img_thr.shape[1]-1):
            ref_pt, img_copy = find_white_points(img_thr, img_copy, ref_pt, w)
            chart_data.append(ref_pt)
        points[idx] = chart_data
        idx += 1

    # cv2.imshow('Image with points', img_copy)
    return points
            
def get_data(ref_len_in_pixel, data_lines, ref_lines, ref_lines_names):
    time_scale = 200/ref_len_in_pixel
    num_pixels = len(data_lines[1])

    time_raw_ms = np.arange(num_pixels) * time_scale
    max_time_ms = time_raw_ms[-1]
    time_scale_ms = np.arange(0.0, max_time_ms + 1.0, 1.0)
    final_data = {'time_ms': time_scale_ms}

    for key, list_data in data_lines.items():
        lead_name = ref_lines_names[key-1]        
        y_raw = np.array(list_data)
        y_normalized = ref_lines[key-1] - y_raw    
        interpolator = interp1d(time_raw_ms, y_normalized, kind='linear', fill_value="extrapolate")        
        y_interpolated = interpolator(time_scale_ms)        
        final_data[lead_name] = y_interpolated

    df = pd.DataFrame(final_data)
    df = df.round().astype(int)
    return df

def plot_signals(df, ref_lines_names):
    fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(12, 16), sharex=True)
    
    for i, lead in enumerate(ref_lines_names):
        axes[i].plot(df['time_ms'], df[lead], color='black', linewidth=1)
        axes[i].set_ylabel(lead, rotation=0, labelpad=20, va='center', fontweight='bold')
        axes[i].grid(True, linestyle='--', alpha=0.5)

        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)

    axes[-1].set_xlabel('Czas [ms]', fontweight='bold')
    axes[-1].spines['bottom'].set_visible(True)
    
    plt.suptitle('Wyekstrahowane sygnały EKG', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96) 
    
    plt.show()

def run_basic_version(input_dir, output_dir):
    ref_paths = glob.glob(os.path.join(input_dir, '*3_3.JPG'))
    img_ref = cv2.imread(ref_paths[0]) if ref_paths else None
    if img_ref is None:
        print(f'BŁĄD: W folderze {ref_paths[0]} nie ma pliku referencyjnego P3_3.JPG')
    ref_scale = get_ref_scale_in_pixel(img_ref)
    
    ref_lines_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    image_files = glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]'))
    
    for img_path in image_files:
        if '3_3.JPG' in img_path:
            continue
            
        filename = os.path.basename(img_path)
        img_test = cv2.imread(img_path)
        if img_test is None: continue

        try:
            img_cropped, _ = get_chart_roi(img_test)
            ref_lines = get_reference_lines(img_cropped)
            if len(ref_lines) < 1: continue
                
            data_lines = find_points(img_cropped, ref_lines)
            df_signals = get_data(ref_scale, data_lines, ref_lines, ref_lines_names)
            
            csv_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.csv')
            df_signals.to_csv(csv_path, index=False)
            print(f"Przetworzono (Basic): {filename}")
            # plot_signals(df_signals, ref_lines_names)
            
        except Exception as e:
            print(f"Błąd dla {filename}: {e}")