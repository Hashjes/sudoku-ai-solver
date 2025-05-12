# train_new_model.py
#!/usr/bin/env python3
import os, shutil, random, glob, time
import cv2, numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ───────── CONFIG ─────────────────────────────
IMG_SIZE      = 28
SYNTH_SAMPLES = 2000
TRAIN_SPLIT   = 0.8
BATCH_SIZE    = 128
EPOCHS        = 30
LR            = 1e-3

RAW_DIR    = 'data/raw_rois'
SYNTH_DIR  = 'synthetic_no_border'
MODEL_DIR  = 'models'
NEW_MODEL  = 'sudoku_digit_model_no_border.keras'
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── CALLBACK PARA TIEMPOS ────────────────────
class TimeLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()
    def on_epoch_end(self, epoch, logs=None):
        print(f"  ↳ Epoch {epoch+1} duró {time.time() - self.start:.1f}s")

# ─── 1) GENERAR SINTÉTICOS SIN MARCO ───────────
def make_synthetic():
    if os.path.exists(SYNTH_DIR):
        shutil.rmtree(SYNTH_DIR)
    for split in ('train','val'):
        for cls in map(str, range(10)):
            os.makedirs(os.path.join(SYNTH_DIR, split, cls), exist_ok=True)

    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]

    for cls in map(str, range(10)):
        imgs = []
        for i in range(SYNTH_SAMPLES):
            img = np.ones((IMG_SIZE, IMG_SIZE), dtype='uint8') * 255
            font = random.choice(fonts)
            scale = random.uniform(0.9, 1.1)
            thickness = random.randint(1, 2)
            (w, h), _ = cv2.getTextSize(cls, font, scale, thickness)
            x = (IMG_SIZE - w)//2 + random.randint(-1,1)
            y = (IMG_SIZE + h)//2 + random.randint(-1,1)
            cv2.putText(img, cls, (x,y), font, scale, (0,), thickness, cv2.LINE_AA)
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((IMG_SIZE/2,IMG_SIZE/2), angle, 1)
            img = cv2.warpAffine(img, M, (IMG_SIZE,IMG_SIZE), borderValue=255)
            _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
            # ¡sin cv2.rectangle!
            imgs.append(img)

        cut = int(SYNTH_SAMPLES * TRAIN_SPLIT)
        for idx, img in enumerate(imgs):
            split = 'train' if idx < cut else 'val'
            fn = f"{cls}_{idx:04d}.png"
            cv2.imwrite(os.path.join(SYNTH_DIR, split, cls, fn), img)
        print(f"[synth] Clase {cls}: {cut} train, {SYNTH_SAMPLES-cut} val")

# ─── 2) AÑADIR ROIs MANUALES ───────────────────
def add_manual_rois():
    for cls in map(str, range(10)):
        rois = glob.glob(os.path.join(RAW_DIR, cls, '*.png'))
        random.shuffle(rois)
        cut = int(len(rois) * TRAIN_SPLIT)
        for p in rois[:cut]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            inv = cv2.bitwise_not(img)
            cv2.imwrite(os.path.join(SYNTH_DIR, 'train', cls, os.path.basename(p)), inv)
        for p in rois[cut:]:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            inv = cv2.bitwise_not(img)
            cv2.imwrite(os.path.join(SYNTH_DIR, 'val', cls, os.path.basename(p)), inv)
        print(f"[manual] Clase {cls}: +{cut} train, +{len(rois)-cut} val")

# ─── 3) GENERADORES CON AUGMENTATION ───────────
def make_generators():
    common = dict(target_size=(IMG_SIZE,IMG_SIZE), color_mode='grayscale', class_mode='sparse', batch_size=BATCH_SIZE)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8,1.2),
        shear_range=5
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory(os.path.join(SYNTH_DIR,'train'), shuffle=True, **common)
    val_gen   = val_datagen.flow_from_directory(os.path.join(SYNTH_DIR,'val'),   shuffle=False, **common)
    return train_gen, val_gen

# ─── 4) MODELO ──────────────────────────────────
def build_model():
    m = Sequential([
        Input((IMG_SIZE,IMG_SIZE,1)),
        Conv2D(32,(3,3),padding='same',activation='relu'),
        MaxPooling2D(), Dropout(0.25),
        Conv2D(64,(3,3),padding='same',activation='relu'),
        MaxPooling2D(), Dropout(0.25),
        Flatten(), Dense(128, activation='relu'), Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    m.compile(optimizer=Adam(LR), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# ─── 5) EJECUCIÓN ──────────────────────────────
if __name__=='__main__':
    print("1) Generando sintéticos sin borde…")
    make_synthetic()
    print("2) Añadiendo manual ROIs…")
    add_manual_rois()
    print("3) Preparando generadores…")
    train_gen, val_gen = make_generators()
    print("4) Construyendo modelo…")
    model = build_model()
    callbacks = [TimeLogger(), EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.2, patience=3)]
    print(f"5) Entrenando hasta {EPOCHS} epochs…")
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)
    out = os.path.join(MODEL_DIR, NEW_MODEL)
    model.save(out)
    print(f"✔ Nuevo modelo guardado en {out}")
    print(f"Final train acc: {history.history['accuracy'][-1]:.4f}")
    print(f"Final   val acc:   {history.history['val_accuracy'][-1]:.4f}")
