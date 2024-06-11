from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
import os
import shutil
import logging
from celery_task_app.tasks import predict_single
from models import Task, Prediction

app = FastAPI()

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}  # допустимые расширения файлов

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/detect/predict", response_model=Task, status_code=202)
async def upload_file(audio_file: UploadFile):
    try:
        # Проверка расширения файла
        if not allowed_file(audio_file.filename):
            raise HTTPException(status_code=400, detail="Invalid file extension. Allowed extensions are: wav, mp3, flac")
        
        # Создаем директорию для сохранения файлов, если ее еще нет
        if not os.path.exists("audio_files"):
            os.makedirs("audio_files")
        
        # Сохраняем загруженный файл
        file_path = os.path.join("audio_files", audio_file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logging.info(f"File saved: {audio_file.filename}")

        task_id = predict_single.delay(file_path)
        return {'task_id': str(task_id), 'status': 'Processing'}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


@app.get('/detect/result/{task_id}', response_model=Prediction, status_code=200,
         responses={202: {'model': Task, 'description': 'Accepted: Not Ready'}})
async def detection_result(task_id: str):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)

    # Проверка на существование task_id
    if task.state == 'PENDING':
        raise HTTPException(status_code=404, detail=f"Task ID {task_id} not found")

    if not task.ready():
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    
    result = task.get()
    return JSONResponse(status_code=200, content={'task_id': str(task_id), 'status': 'Success', 'anlisys_result':result})