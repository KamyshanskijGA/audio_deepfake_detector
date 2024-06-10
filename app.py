from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
import os
import shutil
import logging
from celery_task_app.tasks import predict_single
from models import Task, Prediction

app = FastAPI()


@app.post("/detect/predict", response_model=Task, status_code=202)
async def upload_file(audio_file: UploadFile):
    try:
        # Создаем директорию для сохранения файлов, если ее еще нет
        if not os.path.exists("audio_files"):
            os.makedirs("audio_files")
        
        # Сохраняем загруженный файл
        with open(audio_file.filename, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logging.info(audio_file.filename)
        task_id = predict_single.delay(audio_file.filename)
        return {'task_id': str(task_id), 'status': 'Processing'}
    except Exception as e:
         return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})



@app.get('/detect/result/{task_id}', response_model=Prediction, status_code=200,
         responses={202: {'model': Task, 'description': 'Accepted: Not Ready'}})
async def detection_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():
        print(app.url_path_for('detect'))
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    result = task.get()
    return {'task_id': task_id, 'status': 'Success', 'probability': str(result)}

