# audio_deepfake_detector

API для распознавания подделки голоса. Это приложение распознает является ли голос синтезированнм или он настоящий.

# Архитектура приложения
![image](https://github.com/KamyshanskijGA/audio_deepfake_detector/assets/86370005/6a629854-1544-4945-bd77-5641776f7563)

# Диаграмма последовательности
![image](https://github.com/KamyshanskijGA/audio_deepfake_detector/assets/86370005/91524a7c-cea0-4a48-ab29-51abbc1a5e41)

# Описание запросов API:
1. Загрузка аудиофайла для анализа:
    - Метод: POST
    - URL: /detect/predict
    - Вход: аудиофайл  в одном из этих форматов: 'wav', 'mp3', 'flac'
    - Выход:```{'task_id': str, 'status': 'Processing'}```

2. Получение результата анализа:
    - Метод: GET
    - URL: /detect/result/{task_id}
    - Вход: task_id: str
    - Выход:
        - Если задача в процессе: ```{'task_id': str, 'status': 'Processing'}```
        - Если задача завершена: 
      ```
      {
        'task_id': str, 
        'status': 'Success', 
        'anlisys_result': {
            'probability': float, 
            'predicted_class': str, 
            'metadata': {
              'audio_name': str, 
              'duration': str, 
              'format': str
            }
         }
      }
      ```
В проекте присутствуют следующие ключевые файлы и компоненты:

    app.py - содержит FastAPI приложение для обработки запросов.
    
    models.py - включает модели данных.
    
    tasks.py - содержит задачи Celery для обработки данных.
    
    worker.py - конфигурация Celery worker.
    
    model.py - включает загрузку и использование ML модели.

    model3.ipynb - colab для обучения модели

# Запуск
В директории приложения нужно прописать:
    ```uvicorn app:app```
    ```celery -A celery_task_app.worker worker -l info```
