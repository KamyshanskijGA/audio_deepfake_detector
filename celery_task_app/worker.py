import os
from celery import Celery

# URI для брокера сообщений, в данном случае используется RabbitMQ
BROKER_URI = "pyamqp://guest@localhost//"

# URI для backend, который будет использоваться для хранения результатов задач, в данном случае используется Redis
BACKEND_URI = "redis://localhost:6379/0"

# Инициализация приложения Celery
app = Celery(
    'celery_app',  # Имя приложения Celery
    broker=BROKER_URI,  # URI брокера сообщений
    backend=BACKEND_URI,  # URI backend для хранения результатов
    include=['celery_task_app.tasks']  # Список модулей, в которых Celery будет искать задачи
)