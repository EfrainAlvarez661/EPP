# 🦺 EPP Detector — Streamlit App

**Autor:** Efrain Alvarez  
**Modelo:** YOLOv8s entrenado con dataset PPE Factory (Roboflow)  
**Framework:** Streamlit + Ultralytics

---

## 📌 Descripción

Aplicación web interactiva para la **detección de Equipos de Protección Personal (EPP)** en imágenes y videos en tiempo real, utilizando un modelo YOLOv8 entrenado con el dataset `ppe-factory` de Roboflow.

La app permite:
- Subir imágenes o videos para análisis
- Visualizar las detecciones con bounding boxes
- Consultar métricas del modelo (mAP, precisión, recall)
- Exportar resultados

---

## 🗂️ Estructura del Proyecto

```
epp-detector/
│
├── app.py                        # Aplicación principal Streamlit
├── EPP.ipynb                     # Notebook de entrenamiento y evaluación
├── requirements.txt              # Dependencias del proyecto
├── README.md                     # Este archivo
│
├── models/
│   └── best.pt                   # Pesos del mejor modelo entrenado (YOLOv8s)
│
├── data/
│   └── ppe-factory-1/
│       ├── train/
│       ├── valid/
│       ├── test/
│       └── data.yaml
│
└── runs/
    └── ppe_detector/
        └── weights/
            └── best.pt
```

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/epp-detector.git
cd epp-detector
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run app.py
```

La app estará disponible en `http://localhost:8501`

---

## 📦 Dependencias (`requirements.txt`)

```
streamlit>=1.32.0
ultralytics>=8.0.0
roboflow>=1.1.0
opencv-python-headless>=4.9.0
Pillow>=10.0.0
numpy>=1.24.0
PyYAML>=6.0
torch>=2.0.0
```

---

## 🧠 Modelo

| Parámetro        | Valor                        |
|------------------|------------------------------|
| Arquitectura     | YOLOv8s                      |
| Dataset          | PPE Factory v1 (Roboflow)    |
| Épocas           | 50                           |
| Tamaño de imagen | 416×416                      |
| Batch size       | 16                           |
| Formato export   | PyTorch `.pt` / ONNX         |

### Clases detectadas

Las clases disponibles se definen en `data/ppe-factory-1/data.yaml`. Entre las categorías típicas de EPP se encuentran:

- `helmet` — Casco de seguridad
- `vest` — Chaleco reflectivo
- `gloves` — Guantes
- `boots` — Botas de seguridad
- `no-helmet`, `no-vest`, etc. — Ausencia de EPP

---

## 🚀 Uso de la App

### Detección en imagen
1. Selecciona la pestaña **"Imagen"** en el menú lateral
2. Sube un archivo `.jpg`, `.jpeg` o `.png`
3. Ajusta el umbral de confianza con el slider
4. Haz clic en **"Detectar"** para ver los resultados

### Detección en video
1. Selecciona la pestaña **"Video"**
2. Sube un archivo `.mp4` o `.avi`
3. El modelo procesará cada frame y mostrará las detecciones

### Métricas
- La pestaña **"Métricas"** muestra el rendimiento del modelo: mAP50, mAP50-95, Precisión y Recall

---

## 📓 Notebook de Entrenamiento

El archivo `EPP.ipynb` contiene el pipeline completo de:

1. Instalación de dependencias
2. Descarga del dataset desde Roboflow
3. Exploración y verificación del dataset
4. Lectura de etiquetas (`data.yaml`)
5. Carga del modelo base YOLOv8s
6. Entrenamiento (50 épocas)
7. Evaluación con métricas
8. Inferencia en imágenes y videos de test
9. Exportación a ONNX

> El notebook fue desarrollado en Google Colab. Para ejecutarlo localmente, ajusta las rutas (`/content/...` → ruta local).

---

## 📊 Resultados del Entrenamiento

Los resultados, curvas de entrenamiento y matrices de confusión se guardan automáticamente en:

```
runs/ppe_detector/
```

Para visualizarlos dentro de la app o con TensorBoard:

```bash
tensorboard --logdir runs/ppe_detector
```

---

## 🛠️ Ejemplo de código — `app.py`

```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="EPP Detector", page_icon="🦺", layout="wide")
st.title("🦺 Detección de EPP con YOLOv8")

model = YOLO("models/best.pt")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
confidence = st.slider("Umbral de confianza", 0.1, 1.0, 0.25)

if uploaded_file:
    image = Image.open(uploaded_file)
    results = model.predict(np.array(image), conf=confidence)
    annotated = results[0].plot()
    st.image(annotated, caption="Resultado de detección", use_column_width=True)
```

---

## 👤 Autor

**Efrain Alvarez**  
Proyecto de detección de Equipos de Protección Personal con visión por computadora.

---

## 📄 Licencia

Este proyecto es de uso académico y educativo.
"# EPP" 
