from pydantic import BaseModel

class Task(BaseModel):
    """Представление задачи Celery"""
    task_id: str  # Идентификатор задачи
    status: str  # Статус задачи 

class Metadata(BaseModel):
    """Представление метаданных"""
    audio_name: str  # Имя аудиофайла
    duration: float  # Длительность аудиофайла в секундах
    format: str  # Формат аудиофайла

class Prediction(BaseModel):
    """Результат задачи предсказания"""
    task_id: str  # Идентификатор задачи
    status: str  # Статус задачи
    result: str  # Результат предсказания ('Real' или 'Fake')
    probability: float  # Вероятность предсказания
    metadata: Metadata  # Метаданные аудиофайла