# 🧠 Predicción de la presencia de depresión en estudiantes de la UPAO usando regresión lógica

## 📌 ¿Para qué sirve?

Este proyecto permite predecir la probabilidad de que un estudiante presente síntomas depresivos, usando un modelo de machine learning expuesto a través de una API Flask.

Puede ser utilizado como herramienta de apoyo en investigaciones, aplicaciones educativas o sistemas de monitoreo de salud mental.

---

## 🛠️ Herramientas utilizadas

- **Python 3**
- **Flask** – Para construir la API REST
- **scikit-learn** – Para el modelo de Regresión Logística
- **pandas / numpy** – Para manejo y transformación de datos
- **joblib / pickle** – Para serializar el modelo
- **Flask-CORS** – Para permitir peticiones desde otros orígenes
- **Gunicorn** – Para despliegue en producción
- **Render.com** – Para despliegue opcional
- **matplotlib / seaborn** – Para análisis exploratorio (no en producción)

---

## 📂 Estructura del proyecto
```bash
ml_api/
├── app.py # API principal (Flask)
├── requirements.txt # Librerías necesarias
├── model/ # Modelo, scaler y columnas
│ ├── modelo.pkl
│ ├── scaler.pkl
│ └── columnas_modelo.json
├── utils/
│ └── preprocessing.py # Preprocesamiento de entrada
├── monitoring/ # Monitoreo y logging
│ ├── drift_detection.py
│ ├── log_predictions.py
│ └── log_registros.py
├── retraining/ # Reentrenamiento por drift
│ ├── retrain.py
│ └── retrain_log.txt
├── data/ # Datos de registro en CSV
│ ├── registros.csv
│ └── log_registros.csv
└── render.yaml # Configuración para Render (vacío)

```
## ℹ️ Instalación local
```bash
git clone https://github.com/tu_usuario/IA-DEPRESION
cd ml_api
```
## ⚠️ Crear entorno virtual
```bash
python -m venv venv
venv\Scripts\activate
```
## ℹ️ Instalar dependencias
```bash
pip install -r requirements.txt
```
## 🟢 Ejecutar la app
```bash
python app.py
```
