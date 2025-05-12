# Sudoku AI Solver

## Descripción
Aplicación web que permite subir una imagen de un Sudoku, reconoce automáticamente los dígitos con una CNN entrenada y ofrece la solución completa. Permite corregir manualmente el reconocimiento antes de resolver y muestra la solución final.

## Estructura del proyecto
```
.
├── app.py
├── requirements.txt
├── Procfile
├── runtime.txt (opcional)
├── models/
│   └── sudoku_digit_model_no_border.keras
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/
│   ├── base.html
│   ├── edit.html
│   ├── index.html
│   ├── result.html
│   └── error.html
└── utils.py
```

## Requisitos
- Python 3.10 o superior  
- Flask  
- TensorFlow (Keras)  
- OpenCV (cv2)  
- NumPy  
- Gunicorn (para producción)

## Instalación
1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/sudoku-ai-solver.git
   cd sudoku-ai-solver
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux / macOS
   venv\Scripts\activate       # Windows
   ```
3. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Entrenamiento del modelo
El script `train_new_model.py` genera datos sintéticos, añade ROIs manuales y entrena la CNN:
```bash
python train_new_model.py
```
El modelo entrenado se guardará en `models/sudoku_digit_model_no_border.keras`.

## Uso local
1. Asegúrate de que `app.py` apunta a `models/sudoku_digit_model_no_border.keras`.  
2. Inicia la aplicación:
   ```bash
   flask run
   ```
3. Abre en el navegador `http://127.0.0.1:5000`, sube una imagen y prueba la solución.

## Despliegue
Para producción se recomienda usar un PaaS como Render o Railway:

1. Asegúrate de tener `Procfile`:
   ```
   web: gunicorn app:app --bind 0.0.0.0:$PORT
   ```
2. Añade `requirements.txt` y (opcional) `runtime.txt`.  
3. Sube el repositorio a GitHub y conecta tu proyecto en la plataforma.

  
