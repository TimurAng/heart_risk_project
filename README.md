# 🫀 Heart Risk Prediction Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Flask](https://img.shields.io/badge/Web-Framework-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-yellow)

ML модель для предсказания риска сердечно-сосудистых заболеваний с веб-интерфейсом.

## 📋 Оглавление

- [О проекте](#о-проекте)
- [Функциональность](#функциональность)
- [Установка и запуск](#установка-и-запуск)
- [Использование](#использование)
- [Структура проекта](#структура-проекта)
- [Модель ML](#модель-ml)
- [API endpoints](#api-endpoints)
- [Разработчики](#разработчики)

## 🎯 О проекте

Проект представляет собой систему прогнозирования риска сердечных заболеваний на основе медицинских параметров пациента. Включает в себя:

- **ML модель** - обучена на реальных медицинских данных
- **REST API** - Flask-based веб-сервис
- **Веб-интерфейс** - удобная форма для ввода данных
- **Анализ данных** - Jupyter notebook с исследованием

## ⚡ Функциональность

- ✅ Прогнозирование риска сердечного заболевания (0-1)
- ✅ REST API для интеграции с другими системами
- ✅ Веб-интерфейс для ручного ввода данных
- ✅ Загрузка и обработка CSV файлов
- ✅ Сохранение результатов предсказаний

## 🚀 Установка и запуск

### Предварительные требования

- Python 3.8+
- pip (менеджер пакетов Python)

### 1. Клонирование репозитория
```bash
git clone https://github.com/ваш-username/heart_risk_project.git
cd heart_risk_projectКлонирование репозитория
```
### 2. Создайте и активируйте виртуальное окружение
```bash
# Для macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Для Windows
python -m venv .venv
.venv\Scripts\activate
```
### 3. Установите зависимости
```bash
pip install -r requirements.txt
```
### 4. Запустите веб-сервер
```bash
uvicorn main:app --reload
```
### 5. Откройте приложение в браузере

http://127.0.0.1:8000

После запуска приложения загрузите CSV-файл "heart_test.csv" с данными пациентов и получите предсказания.

