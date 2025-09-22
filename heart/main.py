from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import io
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional
import uuid

# --- 1. Инициализация приложения FastAPI ---
app = FastAPI(
    title="API для предсказания риска сердечного приступа",
    description="Загрузите CSV файл с данными пациентов для получения предсказаний."
)

# Монтируем статические файлы и шаблоны
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Глобальная переменная для хранения результатов
prediction_results = {}

# --- 2. Загрузка модели при старте приложения ---
try:
    model = joblib.load('best_heart_risk_model.joblib')
except FileNotFoundError:
    # Для демонстрации создаем mock модель
    class MockModel:
        def predict_proba(self, X):
            return np.random.rand(X.shape[0], 2)
    model = MockModel()
    print("Используется mock модель для демонстрации")

# Функция предобработки данных
def preprocess_data(data):
    """
    Удаляем столбцы и просто дропаем строки с NaN
    """
    columns_to_drop = [ 
        'id', 
        'CK-MB', 
        'Troponin',
        'Previous Heart Problems', 
        'Medication Use', 
        'Blood sugar',
        'Unnamed: 0'
    ]
    
    for col in columns_to_drop:
        if col in data.columns:
            data = data.drop(col, axis=1)

    if 'Gender' in data.columns:
        le = LabelEncoder()
        data['Gender'] = le.fit_transform(data['Gender'].astype(str))
    
    # Просто удаляем строки с пропущенными значениями
    data = data.dropna()
    
    return data

# --- 3. Главная страница с HTML интерфейсом ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- 4. Endpoint для предсказания ---
@app.post("/predict")
async def create_prediction(file: UploadFile = File(...)):
    try:
        # Чтение данных
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # СОХРАНЯЕМ ID ДО ПРЕДОБРАБОТКИ
        if 'id' in data.columns:
            original_ids = data['id'].copy()
        else:
            original_ids = pd.RangeIndex(1, len(data) + 1)
        
        # Предобработка данных (может удалять строки с NaN)
        processed_data = preprocess_data(data)
        
        # Теперь получаем предсказания только для обработанных данных
        predictions_proba = model.predict_proba(processed_data)[:, 1]
        
        # Соответствуем ID оставшимся строкам
        if len(processed_data) == len(original_ids):
            ids = original_ids
        else:
            if 'id' in data.columns:
                remaining_indices = processed_data.index
                ids = original_ids.iloc[remaining_indices]
            else:
                ids = pd.RangeIndex(1, len(processed_data) + 1)
        
        # Создаем список результатов
        results = []
        for patient_id, proba in zip(ids, predictions_proba):
            results.append({
                'id': str(patient_id),
                'risk_probability': round(float(proba), 4)
            })
        
        # Сохраняем результаты с уникальным ID
        result_id = str(uuid.uuid4())
        prediction_results[result_id] = results
        
        return JSONResponse({
            'success': True,
            'result_id': result_id,
            'results': results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# --- 5. Endpoint для скачивания результатов в CSV ---
@app.get("/download_results/{result_id}")
async def download_results(result_id: str):
    if result_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Результаты не найдены")
    
    results = prediction_results[result_id]
    
    # Создаем CSV в памяти
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Записываем заголовки
    writer.writerow(['id', 'risk_probability'])
    
    # Записываем данные
    for result in results:
        writer.writerow([result['id'], result['risk_probability']])
    
    csv_content = output.getvalue()
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=prediction_results_{result_id[:8]}.csv",
            "Content-Type": "text/csv; charset=utf-8"
        }
    )

# --- 6. Endpoint для получения только результатов (без скачивания) ---
@app.get("/api/results/{result_id}")
async def get_results(result_id: str):
    if result_id not in prediction_results:
        raise HTTPException(status_code=404, detail="Результаты не найдены")
    
    return JSONResponse({
        'success': True,
        'results': prediction_results[result_id]
    })