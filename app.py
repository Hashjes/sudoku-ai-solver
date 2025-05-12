import os
import time
import logging
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from utils import (
    pre_process_image,
    find_corners_of_largest_polygon,
    crop_and_warp,
    infer_grid,
    get_digits_full
)

app = Flask(__name__)
UPLOAD_FOLDER      = os.path.join(app.root_path, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}

def allowed_file(fn):
    return '.' in fn and fn.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

MODEL_PATH       = os.path.join(app.root_path, 'models', 'sudoku_digit_model_no_border.keras')
digit_model      = load_model(MODEL_PATH, compile=False)  # compile=False acelera el arranque
THRESHOLD        = 0.2
MIN_BLACK_PIXELS = 35
TIMEOUT_SECONDS  = 5.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Backtracking solver with timeout and optional step recording
def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, r, c, v):
    if any(board[r][j] == v for j in range(9)): return False
    if any(board[i][c] == v for i in range(9)): return False
    br, bc = 3*(r//3), 3*(c//3)
    for i in range(br, br+3):
        for j in range(bc, bc+3):
            if board[i][j] == v: return False
    return True

def solve_sudoku(board, start, max_seconds, steps=None):
    if time.time() - start > max_seconds:
        return False
    empty = find_empty(board)
    if not empty:
        return True
    r, c = empty
    for v in range(1, 10):
        if is_valid(board, r, c, v):
            board[r][c] = v
            if steps is not None:
                # grab a snapshot after placing v
                steps.append([row[:] for row in board])
            if solve_sudoku(board, start, max_seconds, steps):
                return True
            board[r][c] = 0
            if time.time() - start > max_seconds:
                return False
    return False

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'GET':
        # pantalla inicial con index.html
        return render_template('index.html')

    # POST: carga imagen
    file = request.files.get('sudoku_image')
    if not file or not allowed_file(file.filename):
        return redirect(url_for('index'))

    fn   = secure_filename(file.filename)
    dst  = os.path.join(UPLOAD_FOLDER, fn)
    file.save(dst)
    original_url = url_for('static', filename=f'uploads/{fn}')

    recognized = [[0]*9 for _ in range(9)]
    roi_urls = []

    try:
        gray    = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        proc    = pre_process_image(gray)
        corners = find_corners_of_largest_polygon(proc)
        warped  = crop_and_warp(gray, corners)

        squares = infer_grid(warped)
        digits  = get_digits_full(warped, squares, 28)

        roi_dir = os.path.join(UPLOAD_FOLDER, f"rois_{fn}")
        os.makedirs(roi_dir, exist_ok=True)
        flat = []
        for idx, roi in enumerate(digits):
            path = os.path.join(roi_dir, f"{idx}.png")
            cv2.imwrite(path, roi)
            roi_urls.append(url_for('static', filename=f"uploads/rois_{fn}/{idx}.png"))
            total, white = roi.size, cv2.countNonZero(roi)
            black = total - white
            if black < MIN_BLACK_PIXELS:
                flat.append(0)
            else:
                inv = cv2.bitwise_not(roi)
                inp = inv.reshape(1,28,28,1)/255.0
                p = digit_model.predict(inp, verbose=0)[0]
                pred,conf = int(np.argmax(p)), float(np.max(p))
                flat.append(pred if conf>=THRESHOLD else 0)

        recognized = np.array(flat).reshape(9,9).tolist()

    except Exception as e:
        logging.warning(f"Reconocimiento/warp falló: {e}")

    # Intento de resolver y capturar pasos
    board_copy = [row[:] for row in recognized]
    start = time.time()
    steps = []
    solved = solve_sudoku(board_copy, start, TIMEOUT_SECONDS, steps)

    if solved:
        return render_template('result.html',
                               original_url=original_url,
                               recognized_board=recognized,
                               edited_board=None,
                               solved_board=board_copy,
                               errors=[],
                               roi_urls=roi_urls,
                               steps=steps)
    else:
        return render_template('edit.html',
                               original_url=original_url,
                               recognized_board=recognized,
                               roi_urls=roi_urls)

@app.route('/solve', methods=['POST'])
def solve():
    original_url = request.form['original_url']
    rec = [[int(request.form[f'rec_{i}_{j}']) for j in range(9)] for i in range(9)]

    edited = []
    for i in range(9):
        row = []
        for j in range(9):
            v = request.form.get(f'cell_{i}_{j}','').strip()
            row.append(int(v) if v.isdigit() else 0)
        edited.append(row)

    roi_urls = request.form.getlist('roi_urls')

    start = time.time()
    copyb = [r[:] for r in edited]
    steps = []
    solved = solve_sudoku(copyb, start, TIMEOUT_SECONDS, steps)
    solved_board = copyb if solved else None

    errors = []
    if solved_board:
        for i in range(9):
            for j in range(9):
                if edited[i][j] != 0 and edited[i][j] != solved_board[i][j]:
                    errors.append((i,j))

    return render_template('result.html',
                           original_url=original_url,
                           recognized_board=rec,
                           edited_board=edited,
                           solved_board=solved_board,
                           errors=errors,
                           roi_urls=roi_urls,
                           steps=steps if solved else None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
