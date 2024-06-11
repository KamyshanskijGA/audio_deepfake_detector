import importlib
import logging
from celery import Task

from .worker import app

class PredictTask(Task):
    """
    Абстракция класса задачи Celery для поддержки загрузки ML модели.
    """
    abstract = True

    def __init__(self):
        super().__init__()
        self.model = None  # Инициализация переменной для хранения модели

    def __call__(self, *args, **kwargs):
        """
        Загрузка модели при первом вызове (то есть при первой обработке задачи).
        Избегает необходимости загружать модель при каждом запросе задачи.
        """
        if not self.model:
            logging.info('Loading Model...')
            # Импорт модуля, указанного в path
            module_import = importlib.import_module(self.path[0])
            # Получение объекта модели из модуля
            model_obj = getattr(module_import, self.path[1])
            self.model = model_obj()  # Инициализация объекта модели
            logging.info('Model loaded')
        return self.run(*args, **kwargs)  # Выполнение метода run с переданными аргументами

@app.task(
    ignore_result=False,
    bind=True,
    base=PredictTask,
    path=('celery_task_app.ml.model', 'DetectionModel'),
    name='{}.{}'.format(__name__, 'Detection')
)
def predict_single(self, data):
    """
    По сути, метод run класса PredictTask.
    """
    # Выполнение предсказания с использованием модели
    pred_array = self.model.predict(data)
    positive_prob = pred_array
    return str(positive_prob)  # Возвращение результата предсказания в виде строки