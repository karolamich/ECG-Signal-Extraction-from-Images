import cv2
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from basic_version import process_basic_image

# ------------DETEKCJA PODZIALU PANELI-----------

def detect_panel_split(img):
    split_x = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, img_thr = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('Image threshold', img_thr)    
    
    img_processed = img_thr

    lines = cv2.HoughLinesP(
                img_processed, 
                1, 
                np.pi/180, 
                threshold=50, 
                minLineLength=15,
                maxLineGap=70
                )

    horizontal_lines = []
    tolerance_angle = 0
    img_copy = img.copy()
    max_length = 0
    longest_line = None
    
    for points in lines:
        x1, y1, x2, y2 = points[0]
        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)

        if abs(abs(angle_deg)-90) <= tolerance_angle:
            horizontal_lines.append((y1, (x1, y1, x2, y2)))
            length = np.hypot(x2 - x1, y2 - y1)

            if length > max_length:
                max_length = length
                longest_line = (x1, y1, x2, y2)

    if longest_line is not None:
        x1, y1, x2, y2 = longest_line
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 3)
        split_x = x1

    img_copy_1 = img.copy()
    img_copy_2 = img.copy()

    # cv2.imshow('Image with split line', img_copy)
    
    height, width = img.shape[:2]
    if split_x is None:
        return None, None 

    split_col = int(max(0, min(width, split_x)))

    cropped_img_left = img_copy_1[:, :split_col]
    cropped_img_right = img_copy_2[:, split_col:]

    # cv2.imshow('Cropped left panel', cropped_img_left)
    # cv2.imshow('Cropped right panel', cropped_img_right)

    return cropped_img_left, cropped_img_right


# ------------PRZETWARZANIE WSTEPNE-----------

def auto_deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    lines = cv2.HoughLinesP(
        thresh, 
        1, 
        np.pi/180, 
        threshold=100, 
        minLineLength=100, 
        maxLineGap=10
    )
    
    if lines is None:
        return img 
        
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        if -15 < angle_deg < 15:
            angles.append(angle_deg)
            
    if not angles:
        return img

    median_angle = np.median(angles)
    
    if abs(median_angle) < 0.1:
        return img

    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    rotated_img = cv2.warpAffine(
        img, 
        rotation_matrix, 
        (width, height), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    
    cv2.imshow('Obraz po wyprostowaniu', rotated_img)
    return rotated_img

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray', gray)

    bg = cv2.medianBlur(gray, 41)
    # cv2.imshow('Background', bg)

    diff = cv2.subtract(bg, gray)
    # cv2.imshow('Difference', diff)

    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('Preprocessed', norm)

    blur = cv2.GaussianBlur(norm,(5,5),0)
    # cv2.imshow('Blur', blur)

    _, thr = cv2.threshold(blur, 82, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Threshold', thr)
    return thr


# ------------EKSTRAKCJA SYGNALU I SKALI-----------

def get_grid_scale_px(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    col_sum = thr.sum(axis=0).astype(float)
    col_sum -= col_sum.mean()

    corr = np.correlate(col_sum, col_sum, 'full')
    corr = corr[len(corr) // 2:]

    peaks, _ = find_peaks(corr[1:], height=corr.max() * 0.1, distance=15)
    peaks += 1

    if len(peaks) == 0:
        return max(1, img.shape[1] // 6)

    one_sec_px = int(round(peaks[0]))
    large_sq = max(1, one_sec_px // 5)
  
    return large_sq

def detect_baselines(img_binary, img_rgb, n_channels=7, margin_frac=0.03):
    height, width = img_binary.shape
    margin = int(height * margin_frac)
    strip_h = (height - 2 * margin) // n_channels
 
    baselines = []
    for i in range(n_channels):
        y_start = margin + i * strip_h
        y_end = y_start + strip_h
        strip = img_binary[y_start:y_end, :]
 
        row_sums = strip.sum(axis=1)
 
        if row_sums.max() == 0:
            baselines.append((y_start + y_end) // 2)
            continue
 
        rows = np.arange(len(row_sums))
        weights = row_sums.astype(float)
        threshold = np.percentile(weights, 40)
        weights_filtered = np.where(weights < threshold, 0.0, weights)
 
        if weights_filtered.sum() == 0:
            baselines.append((y_start + y_end) // 2)
            cv2.line(img_rgb, (0, y_start), (width, y_start), (0, 255, 0), 1)
        else:
            center = int(np.average(rows, weights=weights_filtered))
            cv2.line(img_rgb, (0, y_start + center), (width, y_start + center), (0, 0, 255), 1) 
            baselines.append(y_start + center)

    # cv2.imshow('Image with baselines', img_rgb)
    return baselines
 
def find_white_points(img, img_mod, ref_pt, w):
    i = 1
    found = None

    while True:
        ref_pt = np.clip(ref_pt, 0, img.shape[0] - 1)
        if img[ref_pt, w] == 255:
            found = ref_pt
            break
        elif ref_pt + i <= img.shape[0] - 1 and img[ref_pt + i, w] == 255:
            found = ref_pt + i
            break
        elif ref_pt - i >= 0 and img[ref_pt - i, w] == 255:
            found = ref_pt - i
            break
        else:
            i += 1
            if i > img.shape[0]:
                img_mod[ref_pt, w] = (0, 0, 255)
                return ref_pt, img_mod

    top = found
    while top - 1 >= 0 and img[top - 1, w] == 255:
        top -= 1
    bottom = found
    while bottom + 1 <= img.shape[0] - 1 and img[bottom + 1, w] == 255:
        bottom += 1
 
    ref_pt = (top + bottom) // 2
    img_mod[ref_pt, w] = (0, 0, 255)
 
    return ref_pt, img_mod
 
def find_points(img_binary, ref_lines):
    height, width = img_binary.shape
    n = len(ref_lines)

    boundaries = [0]
    for i in range(n - 1):
        mid = (ref_lines[i] + ref_lines[i + 1]) // 2
        boundaries.append(mid)
    boundaries.append(height)
 
    img_mod = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
 
    for y_ref in ref_lines:
        pass
        # cv2.line(img_mod, (0, y_ref), (width, y_ref), (0, 255, 0), 1)
 
    points = {}
    for idx, y_ref in enumerate(ref_lines):
        y_lo = boundaries[idx]
        y_hi = boundaries[idx + 1]
 
        strip = img_binary[y_lo:y_hi, :]
        strip_mod = img_mod[y_lo:y_hi, :]

        ref_pt_local = y_ref - y_lo
 
        chart_data = []
        for w in range(width - 1):
            ref_pt_local, strip_mod = find_white_points(
                strip, strip_mod, ref_pt_local, w)
            chart_data.append(ref_pt_local + y_lo)
 
        points[idx + 1] = chart_data

    # cv2.imshow('Image with points', img_mod)
    return points


# ------------PRZELICZANIE DANYCH I OBLICZANIE BPM-----------

def signals_to_dataframe(signals_left, signals_right,baselines_left, baselines_right, large_sq_px, channel_names):
    ms_per_px = 200.0 / large_sq_px
 
    all_signals = {}
 
    for i, (ch_name, baseline) in enumerate(zip(channel_names[:6], baselines_left)):
        y_raw = np.array(signals_left[i + 1], dtype=float)
        all_signals[ch_name] = baseline - y_raw
 
    for i, (ch_name, baseline) in enumerate(zip(channel_names[6:], baselines_right)):
        y_raw = np.array(signals_right[i + 1], dtype=float)
        all_signals[ch_name] = baseline - y_raw
 
    n_px = min(len(signals_left[1]), len(signals_right[1]))
    time_raw = np.arange(n_px) * ms_per_px
    time_1ms = np.arange(0.0, time_raw[-1] + 1.0, 1.0)
 
    final = {'time_ms': time_1ms}
    for ch_name in channel_names:
        sig = all_signals[ch_name][:n_px]
        t_sig = np.arange(len(sig)) * ms_per_px
        interp_fn = interp1d(t_sig, sig, kind='linear', fill_value='extrapolate')
        final[ch_name] = np.round(interp_fn(time_1ms)).astype(int)
 
    return pd.DataFrame(final)
 
def compute_bpm(df, lead='II', ms_per_sample=1.0):
    if lead not in df.columns:
        lead = df.columns[1]
 
    signal = df[lead].values.astype(float)
    sig_range = signal.max() - signal.min()
    if sig_range == 0:
        return None
 
    sig_norm = (signal - signal.mean()) / sig_range
    min_distance = int(300 / ms_per_sample)
 
    peaks, _ = find_peaks(sig_norm, height=0.3, distance=min_distance)
 
    if len(peaks) < 2:
        return None
 
    mean_rr = np.mean(np.diff(peaks)) * ms_per_sample
    return round(60000.0 / mean_rr, 1)
 
def plot_signals(df, channel_names, bpm=None, filename=""):
    num_channels = len(channel_names)
    
    fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(12, 1.5 * num_channels), sharex=True)
    
    if num_channels == 1:
        axes = [axes]
        
    for i, lead in enumerate(channel_names):
        if lead in df.columns:
            axes[i].plot(df['time_ms'], df[lead], color='black', linewidth=1)
            axes[i].set_ylabel(lead, rotation=0, labelpad=25, va='center', fontweight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.5)

            axes[i].spines['top'].set_visible(False)
            axes[i].spines['right'].set_visible(False)
            axes[i].spines['bottom'].set_visible(False)

    axes[-1].set_xlabel('Czas [ms]', fontweight='bold')
    axes[-1].spines['bottom'].set_visible(True)
    
    title = 'Wyekstrahowane sygnały EKG'
    if filename:
        title += f' ({filename})'
        
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.96) 
    
    plt.show()


# ------------GLOWNA FUNKCJA-----------
 
CHANNEL_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
 
def process_scan_image(img_path, output_dir):
    filename = os.path.basename(img_path)
    print(f"\nPrzetwarzanie: {filename}")
 
    img = cv2.imread(img_path)
    if img is None:
        print(f"BŁĄD: Nie można wczytać obrazu.")
        return False
 
    left_panel, right_panel = detect_panel_split(img)
    if left_panel is None:
        print("UWAGA: Nie wykryto podziału - podział w połowie obrazu.")
        mid = img.shape[1] // 2
        left_panel = img[:, :mid].copy()
        right_panel = img[:, mid:].copy()

 
    left_panel = auto_deskew(left_panel)
    right_panel = auto_deskew(right_panel)
 
    large_sq_px = get_grid_scale_px(left_panel)
 
    left_bin = preprocess_img(left_panel)
    right_bin = preprocess_img(right_panel)
 
    baselines_left = detect_baselines(left_bin, left_panel.copy(), n_channels=6)
    baselines_right = detect_baselines(right_bin, right_panel.copy(), n_channels=6)

    signals_left  = find_points(left_bin,  baselines_left)
    signals_right = find_points(right_bin, baselines_right)
 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
 
    df = signals_to_dataframe(
        signals_left, signals_right,
        baselines_left, baselines_right,
        large_sq_px, CHANNEL_NAMES
    )
 
    bpm = compute_bpm(df, lead='II')
    if bpm is not None:
        print(f"Wykryte tętno: {bpm} BPM")
    else:
        print("Nie udało się wyliczyć tętna.")
 
    csv_name = os.path.splitext(filename)[0] + '.csv'
    csv_path = os.path.join(output_dir, csv_name)
    df.to_csv(csv_path, index=False)

    plot_signals(df, CHANNEL_NAMES, bpm=bpm, filename=filename)
    return True

def run_extended_version(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
 
    patterns = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    image_files = []
    for pat in patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pat)))
    image_files = sorted(set(image_files), key=str.lower)
    print(image_files)
 
    if not image_files:
        print(f"Brak obrazów w folderze: {input_dir}")
        return
 
    print(f"Znaleziono {len(image_files)} plik(ów) do przetworzenia.")
    for img_path in image_files:
        try:
            if img_path == os.path.join(input_dir, 'ekg_photo.jpg'):
                process_scan_image(img_path, output_dir)
            else:
                print(f"Przetwarzanie obrazu: {img_path}")
                ref_scale = 226
                df, bpm = process_basic_image(img_path, ref_scale, output_dir)
                if df is not None:
                    print(f"Wykryte tętno: {bpm} BPM")

        except Exception as e:
            print(f"Błąd dla {os.path.basename(img_path)}: {e}")
 
    print(f"\nZakończono")


