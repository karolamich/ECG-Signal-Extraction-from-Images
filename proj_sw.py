import cv2
import numpy as np

# WCZYTANIE OBRAZÓW
# img_test = cv2.imread('../img/P3_1.JPG')
img_test = cv2.imread('../img/P5_1.JPG')
img_test = cv2.resize(img_test, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
# cv2.imshow('Image test', img_test) 

img_ref = cv2.imread('../img/P3_3.JPG')
img_ref = cv2.resize(img_ref, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
# cv2.imshow('Image ref', img_ref) 

# ------------PRZELICZANIE DLUGOSCI REFERENCYJNEJ OSI X-----------

def get_ref_len_axis_x(img):
    tolerance_angle = 5
    img_ref = img
    img_ref_gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image ref gray', img_ref_gray)

    # img_ref_adaptiv_thr = cv2.adaptiveThreshold(img_ref_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, img_ref_thr = cv2.threshold(img_ref_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image ref gray threshold', img_ref_thr)

    # kernel = np.ones((5, 5), np.uint8)
    # morph = cv2.erode(img_ref_thr,kernel,iterations = 1)
    # cv2.imshow('Image morph', morph)

    width = img_ref.shape[1]
    width_low = int(0.4*width)
    width_up = int(0.6*width)

    roi = img_ref_gray[:, width_low: width_up]
    # cv2.imshow('ROI', roi)
    edges = cv2.Canny(roi,50,150,apertureSize=3)
    vertical_lines =[]
    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=200, # Min number of votes for valid line
                minLineLength=600, # Min allowed length of line
                maxLineGap=5 # Max allowed gap between line for joining them
                )
    print(len(lines))
    x_list = []
    for points in lines:
        x1,y1,x2,y2=points[0]
        # print(x1, y1, x2, y2)

        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        print(angle_deg)
        if abs(abs(angle_deg)-90) <= tolerance_angle:
            rand_col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.line(img_ref,(width_low+x1,y1),(width_low+x2,y2),rand_col,1)
            x_list.append(x1)
            vertical_lines.append([(x1,y1),(x2,y2)])

    x_list.sort()
    x_min = np.mean(x_list[:2])
    x_max = np.mean(x_list[-2:])

    print(x_min, x_max)
    ref_len_in_pixel = x_max - x_min
    print(ref_len_in_pixel)
    # cv2.imshow('Detected lines',img_ref)

    return ref_len_in_pixel


def get_rect_limit(img):
    img_copy = img.copy()
    height, width = img.shape[0] , img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image1', img)
    _, img_thr = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image2', img_thr)
    # blur = cv2.GaussianBlur(img_thr, (3, 3), 0)
    # cv2.imshow('image3', blur)


    lines = cv2.HoughLinesP(
            img_thr, # Input edge image
            1, # Distance resolution in pixels
            np.pi/180, # Angle resolution in radians
            threshold=600, # Min number of votes for valid line
            minLineLength=4*height//5, # Min allowed length of line
            maxLineGap=height//2 # Max allowed gap between line for joining them
            )
    # print(lines)

    horizontal_lines = []
    vertical_lines = []
    tolerance_angle = 2

    if len(lines) > 0:
        for points in lines:
            x1,y1,x2,y2=points[0]
            angle_rad = np.arctan2(y2-y1, x2-x1)
            angle_deg = np.degrees(angle_rad)
            abs_angle = np.abs(angle_deg)

            if abs_angle <= tolerance_angle or abs(180-abs_angle) <= tolerance_angle :
                # cv2.line(img_copy,(x1,y1),(x2,y2),(0, 255, 0),1)
                # print(f'Y1: {y1}, Y2: {y2}')
                horizontal_lines.append([(x1,y1),(x2,y2)])
                
            elif abs(abs_angle-90) <= tolerance_angle:
                # cv2.line(img_copy,(x1,y1),(x2,y2),(0, 0, 255),1)
                vertical_lines.append([(x1,y1),(x2,y2)])
    else:
        print("No lines detected")

    max_line = 0
    pt_y = None
    for line in vertical_lines:        
        if line[0][0] < width//2 and line[1][0] < width//2 and line[0][0] > max_line and line[1][0] > max_line:
            max_line = max(line[0][0], line[1][0])
            pt_y = line

    max_line_x = 0
    pt_x = None
    for line in horizontal_lines:
        if line[0][1] < height//8 and line[1][1] < height//8 and line[0][1] > max_line_x and line[1][1] > max_line_x:
            print(line)
            max_line_x = max(line[0][1], line[1][1])
            pt_x = line

    # cv2.line(img_copy, pt_y[0], pt_y[1], (0, 0, 255), 1)
    # cv2.line(img_copy, pt_x[0], pt_x[1], (0, 255, 0), 1)
    # cv2.imshow('image', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cropped_img = img_copy[pt_x[0][1]+1:height-int(height/20), pt_y[0][0]+1:]
    # cv2.imshow('Cropped image', cropped_img)

    return cropped_img, pt_y[0][0], pt_x[0][1]


def extract_chart_by_density(img):
    # img = cv2.imread(image_path)
    cv2.imshow('Oryginalny obraz', img)
    if img is None:
        print("Nie można wczytać obrazu.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    is_black = gray < 15
    row_black_ratio = np.mean(is_black, axis=1)
    col_black_ratio = np.mean(is_black, axis=0)

    threshold = 0.85
    valid_rows = np.where(row_black_ratio > threshold)[0]
    valid_cols = np.where(col_black_ratio > threshold)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        print("Nie udało się jednoznacznie określić granic wykresu.")
        return

    y1, y2 = valid_rows[0], valid_rows[-1]
    x1, x2 = valid_cols[0], valid_cols[-1]

    cropped_img = img[y1:y2, x1:x2]
    
    # Wyświetlenie wyniku
    # cv2.imshow('Czysty Wykres', cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img, (x1, y1, x2, y2)

# img_cropped, bounds = extract_chart_by_density('../img/P3_1.JPG') 


def get_horizontal_lines(img):
    tolerance_angle = 3
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image ref gray', img_gray)

    _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image gray threshold', img_thr)

    # kernel = np.ones((3, 3), np.uint8)
    # morph = cv2.dilate(img_thr,kernel,iterations = 1)
    # cv2.imshow('Image morph', morph)

    
    width = img_ref.shape[1]
    # width_low = int(0.4*width)
    width_up = int(0.1*width)
    print('Width up:', width_up)

    edges = cv2.Canny(img_thr, 50, 150)
    # cv2.imshow('Canny edges', edges)

    img_processed = edges

    roi = img_processed[:, 0: width_up]
    # cv2.imshow('ROI', roi)


    lines = cv2.HoughLinesP(
                roi, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=50, # Min number of votes for valid line
                minLineLength=40, # Min allowed length of line
                maxLineGap=30 # Max allowed gap between line for joining them
                )

    horizontal_lines = []

    for points in lines:
        x1,y1,x2,y2=points[0]

        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        abs_angle = np.abs(angle_deg)

        if abs_angle <= tolerance_angle or abs(180-abs_angle) <= tolerance_angle :
            cv2.line(img_copy,(x1,y1),(x2,y2),(0, 255, 0),1)
            horizontal_lines.append((y1, (x1, y1, x2, y2)))


    # print('Len of horizontal lines:', len(horizontal_lines))      
    ys = np.array([line[0] for line in horizontal_lines])

    # lines_points = [line[1] for line in horizontal_lines]
    ys.sort()
    print('Ys:', ys)

    hist, bins = np.histogram(ys, bins=range(0, img_gray.shape[0], 50))

    print('Histogram:', hist)
    print('Bins:', bins)
    ref_lines = []
    for i in range(len(bins)-1):
        accumulated_count = sum(hist[:i+1])

        if hist[i] > 0: 
            # print(f'hist[{i}] = {hist[i]}')
            # print(f'Accumulated count up to bin {i}: {accumulated_count}')
            elem = ys[accumulated_count-hist[i]:accumulated_count]

            y_mean = int(np.floor(np.mean(elem)))
            # print(f'Elements in bin {i}: {elem}, mean: {y_mean}')

            ref_lines.append(y_mean)
    print('Reference lines:', ref_lines)
    # cv2.imshow('Image with lines', img_copy)
    # cv2.waitKey(0)

    if len(ref_lines) != 12:
        previous_line = ref_lines[0]
        print("Nie wykryto 12 linii referencyjnych. Wykryto:", len(ref_lines))
        for i in range(len(ref_lines)-1):
            print(i)
            if abs(ref_lines[i]-previous_line) < 40 and i > 0:
                ref_lines[i] = (previous_line+ref_lines[i])//2
                ref_lines.remove(ref_lines[i-1])
            previous_line = ref_lines[i]
        
    for line in ref_lines:
        cv2.line(img_copy, (0, line), (100, line), (0, 0, 255), 1)
    
    # cv2.imshow('Image with lines', img_copy)
    return ref_lines

def find_white_points(img, img_mod, ref_pt, w):
    i = 1
    # print(img.shape[0])
    while True:
        if ref_pt > 0 and ref_pt <= img.shape[0]-1:
            ref_pt = np.clip(ref_pt, 0, img.shape[0]-1)
            print(ref_pt, w)
            if  ref_pt+i <= img.shape[0]-1 and img[ref_pt+i, w] == 255 and ref_pt+i < img.shape[0]:
                img_mod[ref_pt+1, w] = (0, 0, 255)
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


def find_points(img, ref_lines, names):
    points = {}
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image gray threshold', img_thr)
    print(img_thr.shape)

    img_copy = cv2.cvtColor(img_thr, cv2.COLOR_GRAY2BGR)
    for y_ref in ref_lines:
        chart_data = []
        ref_pt = y_ref
        # cv2.line(img_copy, (0, y_ref), (img_thr.shape[1], y_ref), (255, 0, 0), 1)
        for w in range(img_thr.shape[1]-1):
            # if ref_pt < 0 or ref_pt >= img_thr.shape[0]-1:
            #     print(f'Reference point {ref_pt} is out of bounds for width {w}. Skipping this point.')
            #     ref_pt = np.clip(ref_pt, 0, img_thr.shape[0]-1)
            #     continue
            ref_pt, img_copy = find_white_points(img_thr, img_copy, ref_pt, w)
            # if img_thr[ref_pt+1, w] == 255:
            #     img_copy[ref_pt+1, w] = (0, 0, 255)
            #     ref_pt += 1
            # elif img_thr[ref_pt, w] == 255:
            #     img_copy[ref_pt, w] = (0, 0, 255)
            # elif img_thr[ref_pt-1, w] == 255:
            #     img_copy[ref_pt-1, w] = (0, 0, 255)
            #     ref_pt -= 1
            # else:
            #     print(f'No point found at y={y_ref}, w={w}')
         
    cv2.imshow('Image with points', img_copy)
            

# def find_points(img, ref_lines, names=None):
#     # Przechowuje znalezione punkty dla każdej linii referencyjnej
#     points = {} 
    
#     # Tworzymy kopię ORYGINALNEGO obrazu do rysowania
#     img_copy = img.copy()
    
#     # Przetwarzanie obrazu
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    
#     height, width = img_thr.shape

#     # Przechodzimy po 2 pierwszych liniach referencyjnych
#     for i, y_ref in enumerate(ref_lines[:2]):
        
#         # Zabezpieczenie przed wyjściem poza obraz
#         if y_ref < 1 or y_ref >= height - 1:
#             print(f"Linia y={y_ref} jest zbyt blisko krawędzi obrazu!")
#             continue
            
#         found_points_on_line = []
        
#         # Rysowanie niebieskiej linii referencyjnej na obrazie podglądowym
#         # cv2.line(img_copy, (0, y_ref), (width, y_ref), (255, 0, 0), 1)
        
#         # Przeszukiwanie w poziomie (po szerokości)
#         for w in range(width):
#             # Sprawdzamy y_ref oraz piksele sąsiadujące w pionie (wartość 0 = czarny)
#             if img_thr[y_ref, w] == 255:
#                 y_point = y_ref
#             elif img_thr[y_ref + 1, w] == 255:
#                 y_point = y_ref + 1
#             elif img_thr[y_ref - 1, w] == 255:
#                 y_point = y_ref - 1
#             else:
#                 continue # Przechodzimy do następnego piksela, jeśli tu nic nie ma
            
#             # Zaznaczamy znaleziony punkt na czerwono
#             img_copy[y_point, w] = (0, 0, 255)
#             # Zapisujemy koordynaty punktu (x, y)
#             found_points_on_line.append((w, y_point))
        
#         # Zapisujemy punkty dla danej linii pod kluczem (np. nazwą lub indeksem)
#         key = names[i] if names and i < len(names) else f"Line_{y_ref}"
#         points[key] = found_points_on_line
                
#     cv2.imshow('Image with points', img_copy)
#     cv2.waitKey(0)




ref_lines_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# ref_scale = get_ref_len_axis_x(img_ref)

img_cropped, x1, y1 = get_rect_limit(img_test)
# img_cropped, bounds = extract_chart_by_density(img_test) 
ref_lines = get_horizontal_lines(img_cropped)

fiinal_img = img_cropped.copy()
find_points(img_cropped, ref_lines, ref_lines_names)


cv2.waitKey(0)
cv2.destroyAllWindows()