# utils.py
import cv2
import numpy as np

# ──────────────────────────────────────────────────
#         BLOQUE DE PARÁMETROS A AJUSTAR
# ──────────────────────────────────────────────────

# Tamaño del kernel (k×k) para el closing. Usar impares: 1,3,5,...
CLOSING_KERNEL_SIZE = 2

# Número de veces que aplicar dilate tras el adaptiveThreshold
DILATE_ITERATIONS   = 1

# Porcentaje de ampliación del bbox antes de recortar (0.0–1.0)
PAD_RATIO           = 0.4

# Porcentaje del mayor lado del dígito para margin al centrar (0.0–0.5)
MARGIN_RATIO        = 0.1

# ──────────────────────────────────────────────────
#        NO TOCAR A PARTIR DE AQUÍ (salvo que sepas)
# ──────────────────────────────────────────────────

DILATE_KERNEL = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]], np.uint8)

def pre_process_image(img, skip_dilate=False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proc = cv2.GaussianBlur(img, (9,9), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    proc = clahe.apply(proc)
    proc = cv2.adaptiveThreshold(
        proc, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 5
    )
    if not skip_dilate:
        for _ in range(DILATE_ITERATIONS):
            proc = cv2.dilate(proc, DILATE_KERNEL, iterations=1)
    return proc

def remove_grid_lines(img):
    h, w = img.shape
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (w//9, 1))
    vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h//9))
    no_h = cv2.morphologyEx(img, cv2.MORPH_OPEN, horiz)
    no_v = cv2.morphologyEx(no_h, cv2.MORPH_OPEN, vert)
    return cv2.subtract(img, no_v)

def find_corners_of_largest_polygon(img):
    cnts, _ = cv2.findContours(img.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for cnt in cnts:
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts  = approx.reshape(4, 2)
            rect = np.zeros((4,2), dtype="float32")
            s    = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff   = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect
    raise ValueError("No se encontró cuadrilátero de 4 esquinas")

def crop_and_warp(img, crop_rect):
    src  = np.array(crop_rect, dtype='float32')
    side = max(np.linalg.norm(src[0]-src[1]),
               np.linalg.norm(src[1]-src[2]),
               np.linalg.norm(src[2]-src[3]),
               np.linalg.norm(src[3]-src[0]))
    dst  = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype='float32')
    m    = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, m, (int(side), int(side)))
    return cv2.resize(warp, (360,360), interpolation=cv2.INTER_AREA)

def infer_grid(img):
    squares = []
    side = img.shape[0] / 9
    for j in range(9):
        for i in range(9):
            p1 = (i*side, j*side)
            p2 = ((i+1)*side, (j+1)*side)
            squares.append((p1, p2))
    return squares

def cut_from_rect(img, rect):
    (x1,y1), (x2,y2) = rect
    return img[int(y1):int(y2), int(x1):int(x2)]

def scale_and_centre(img, size, margin=0, background=255):
    h, w = img.shape[:2]
    def centre_pad(l):
        s1 = (size-l)//2; s2 = s1 + ((size-l)%2)
        return s1, s2

    if h > w:
        ratio = (size-margin)/h
        nh = size-margin; nw = int(w*ratio)
        lp, rp = centre_pad(nw); tp, bp = margin//2, margin//2
    else:
        ratio = (size-margin)/w
        nw = size-margin; nh = int(h*ratio)
        tp, bp = centre_pad(nh); lp, rp = margin//2, margin//2

    img = cv2.resize(img, (nw, nh))
    img = cv2.copyMakeBorder(img, tp, bp, lp, rp,
                             cv2.BORDER_CONSTANT, None, background)
    return cv2.resize(img, (size, size))

def extract_digit_full(img, rect, size):
    digit = cut_from_rect(img, rect)

    # limpieza inicial fija
    h, w = digit.shape
    pad0 = 6
    if h > 2*pad0 and w > 2*pad0:
        digit = digit[pad0:h-pad0, pad0:w-pad0]

    # binarizar
    _, digit = cv2.threshold(digit, 0, 255,
                             cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    # aplicar closing con el kernel configurado
    kk = CLOSING_KERNEL_SIZE if CLOSING_KERNEL_SIZE % 2 else CLOSING_KERNEL_SIZE+1
    closing_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk,kk))
    digit = cv2.morphologyEx(digit, cv2.MORPH_CLOSE, closing_k, iterations=1)

    # encontrar contorno principal
    cnts, _ = cv2.findContours(digit.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((size, size), np.uint8)
    c = max(cnts, key=cv2.contourArea)
    x, y, wc, hc = cv2.boundingRect(c)

    # pad proporcional al bbox
    pad = int(max(wc, hc) * PAD_RATIO)
    x0 = max(x - pad, 0)
    y0 = max(y - pad, 0)
    x1 = min(x + wc + pad, digit.shape[1])
    y1 = min(y + hc + pad, digit.shape[0])
    roi = digit[y0:y1, x0:x1]

    # margin dinámico para centrar
    margin = max(int(max(wc, hc) * MARGIN_RATIO), 4)
    return scale_and_centre(roi, size, margin=margin)

def get_digits_full(img, squares, size):
    proc    = pre_process_image(img, skip_dilate=True)
    no_grid = remove_grid_lines(proc)
    return [extract_digit_full(no_grid, sq, size) for sq in squares]
