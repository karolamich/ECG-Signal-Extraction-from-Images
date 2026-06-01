import cv2
import numpy as np

# WCZYTANIE OBRAZÓW
img_test = cv2.imread('img/P3_1.JPG')
img_test = cv2.resize(img_test, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
# cv2.imshow('Image test', img_test) 

img_ref = cv2.imread('img/P3_3.JPG')
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
    cv2.imshow('Detected lines',img_ref)

    return ref_len_in_pixel

# get_ref_len_axis_x(img_ref)

def extract_chart_by_density(image_path):
    # 1. Wczytanie obrazu
    img = cv2.imread(image_path)
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

    cv2.imwrite('wykres_koncowy.png', cropped_img)
    
    # Wyświetlenie wyniku
    # cv2.imshow('Czysty Wykres', cropped_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_img

extract_chart_by_density('img/P3_1.JPG')

def get_rectangular_contours(img):
    img_copy = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

    # corners = cv2.goodFeaturesToTrack(
    #     img_gray,
    #     maxCorners=27,
    #     qualityLevel=0.01,
    #     minDistance=10,
    #     blockSize=3,
    #     useHarrisDetector=False,
    #     k=0.04
    # )
    # corners = np.intp(corners)  # Convert to integer coords
    # for corner in corners:
    #     x, y = corner.ravel()
    #     cv2.circle(img_copy, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

    #  ---------------------

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([3, 3, 3])
    # upper_black = np.array([5, 5, 5])

    # Tworzenie binarnej maski (białe piksele tam, gdzie obraz jest czarny)
    mask = cv2.inRange(img, lower_black, upper_black)

    # 3. Operacje morfologiczne (Zamknięcie)
    # Używamy dużego jądra (kernel), aby "załatać" dziury w masce powstałe 
    # przez jasne linie wykresów i teksty na czarnym tle.
    kernel = np.ones((23, 23), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Mask', mask_closed)

    # 4. Szukanie konturów na podstawie zamkniętej maski
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Nie znaleziono żadnych konturów.")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_img = img[y:y+h, x:x+w]
    # print(f"Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    # for contour in contours:
    #     # largest_contour = max(contours, key=cv2.contourArea)
    #     x, y, w, h = cv2.boundingRect(contour)
    #     cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cropped_img = img[y:y+h, x:x+w]

    # Zapis i wyświetlenie wyników
    cv2.imshow('Wykresy', cropped_img)
    
    # Podgląd (wciśnij dowolny klawisz, aby zamknąć okna)
    # cv2.imshow('Oryginal', img_copy)
    # cv2.imshow('Maska (po morfologii)', cv2.resize(mask_closed, (800, 600))) # Zmniejszone dla podglądu
    # cv2.imshow('Wyciety fragment', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------

    # kernel = np.ones((5, 5), np.uint8)
    # morph = cv2.dilate(img_thr,kernel,iterations = 1)
    # contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # rectangular_contours = []
    # for contour in contours:
    #     epsilon = 0.02 * cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, epsilon, True)

    #     if len(approx) == 4:
    #         rectangular_contours.append(approx)

    # for contour in rectangular_contours:
    #     cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    # cv2.imshow('Image with contours', img_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# get_rectangular_contours(img_test)

def get_rect_limit(img):
    img_copy = img.copy()
    height, width = img.shape[0] , img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image1', img)
    _, img_thr = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image2', img_thr)
    blur = cv2.GaussianBlur(img_thr, (3, 3), 0)
    cv2.imshow('image3', blur)


    lines = cv2.HoughLinesP(
            blur, # Input edge image
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
                cv2.line(img_copy,(x1,y1),(x2,y2),(0, 255, 0),1)
                horizontal_lines.append([(x1,y1),(x2,y2)])
                
            elif abs(abs_angle-90) <= tolerance_angle:
                cv2.line(img_copy,(x1,y1),(x2,y2),(0, 0, 255),1)
                vertical_lines.append([(x1,y1),(x2,y2)])
    else:
        print("No lines detected")

    cv2.imshow('image', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# get_rect_limit(img_test)

def get_vertical_lines(img):
    tolerance_angle = 5
    img = img_ref
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Image ref gray', img_gray)

    _, img_thr = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)
    # cv2.imshow('Image gray threshold', img_thr)

    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.dilate(img_thr,kernel,iterations = 1)
    cv2.imshow('Image morph', morph)

    img_processed = img_thr
    width = img_ref.shape[1]
    # width_low = int(0.4*width)
    width_up = int(0.1*width)
    print(width_up)

    roi = img_processed[:, 0: width_up]
    cv2.imshow('ROI', roi)
    edges = cv2.Canny(roi,50,150,apertureSize=3)
    cv2.imshow('Canny edges', edges)

    lines = cv2.HoughLinesP(
                edges, # Input edge image
                1, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians
                threshold=100, # Min number of votes for valid line
                minLineLength=40, # Min allowed length of line
                maxLineGap=25 # Max allowed gap between line for joining them
                )

    horizontal_lines = []
    vertical_lines = []
    y_lines = []
    for points in lines:
        x1,y1,x2,y2=points[0]
        # print(x1, y1, x2, y2)

        angle_rad = np.arctan2(y2-y1, x2-x1)
        angle_deg = np.degrees(angle_rad)
        abs_angle = np.abs(angle_deg)
        # print(abs_angle)
        if abs_angle <= tolerance_angle or abs(180-abs_angle) <= tolerance_angle :
            # rand_col = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            rand_col = (0, 255, 0)
            cv2.line(img,(x1,y1),(x2,y2),rand_col,1)
        #     x_list.append(x1)
            horizontal_lines.append([(x1,y1),(x2,y2)])
        elif abs(abs_angle-90) <= tolerance_angle:
            rand_col = (0, 0, 255)
            # cv2.line(img,(x1,y1),(x2,y2),rand_col,1)
            vertical_lines.append([(x1,y1),(x2,y2)])


    print('Len of horizontal lines:', len(horizontal_lines))
    print('Len of vertical lines:', len(vertical_lines))

    cv2.imshow('Imaage with lines', img)
    cv2.waitKey(0)

cv2.waitKey(0)
# get_vertical_lines(img_ref)
cv2.destroyAllWindows()