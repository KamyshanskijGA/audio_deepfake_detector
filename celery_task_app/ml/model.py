import joblib
from keras.models import load_model
import numpy as np
from keras.initializers import Orthogonal
import logging
import os
import librosa
import soundfile as sf

# MODEL_PATH = os.environ['MODEL_PATH']
MODEL_PATH = rf'/mnt/c/Users/390/Documents/GitHub/audio_deepfake_detector/celery_task_app/ml/model_fin_bin_full_test05 (1).h5'
target_sr = 22050

class DetectionModel:
    """Класс-обёртка для загрузки и использования предобученной модели."""

    def __init__(self):
        # Инициализация модели при создании объекта класса
        self.model = self._load_model_from_path(MODEL_PATH)

    @staticmethod
    def _load_model_from_path(path):
        # Загрузка модели из указанного пути
        # model = joblib.load(MODEL_PATH)
        model = load_model(MODEL_PATH)
        return model

    def extract_features(self, audio_file):
        # Функция для извлечения признаков из аудиофайла
        features = []
        try:
            # Загрузка аудиофайла с заданной частотой дискретизации
            y, sr = librosa.load(audio_file, sr=target_sr)

            # Нормализация громкости
            y = librosa.util.normalize(y)

            # Определение параметров для разделения на фреймы
            frame_length = 2048
            hop_length = 512

            # Извлечение мел-кепстральных коэффициентов (MFCC)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfcc, axis=1)
            mfccs_std = np.std(mfcc, axis=1)

            # Извлечение спектрограммы
            stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
            spectrogram = np.abs(stft)
            stfts_mean = np.mean(spectrogram, axis=1)
            stfts_std = np.std(spectrogram, axis=1)

            # Извлечение звуковой энергии
            energy = librosa.feature.rms(y=y)
            rmss_mean = np.mean(energy)
            rmss_std = np.std(energy)

            # Извлечение Zero-crossing rate (частоты пересечений нуля)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
            zcrs_mean = np.mean(zcr)
            zcrs_std = np.std(zcr)

            # Объединение всех признаков в один массив
            features.append(np.hstack((mfccs_mean, mfccs_std, stfts_mean, stfts_std, rmss_mean, rmss_std, zcrs_mean, zcrs_std)))

        except Exception as e:
            # Обработка ошибок при извлечении признаков
            print(f"Error processing {audio_file}: {str(e)}")
        
        # Добавление дополнительного измерения для совместимости с моделью
        features = np.expand_dims(features, axis=2)
        return features

    def predict(self, data):
        """
        Выполнение предсказания на основе списка предварительно обработанных признаков.
        """

        # Извлечение признаков из аудиофайла
        features = self.extract_features(data)
        # Получение предсказания модели
        prediction = self.model.predict(features)

        # Определение длительности аудиофайла
        duration = librosa.get_duration(filename=str(data))

        # Использование библиотеки soundfile для получения дополнительной метаинформации
        with sf.SoundFile(data) as f:
            format_info = f.format

        # Определение класса на основе вероятности
        if prediction[0][0] < 0.5:
            predicted_class = 'Fake'
            prediction_probability = 1 - prediction[0][0]
        else:
            predicted_class = 'Real'
            prediction_probability = prediction[0][0]

        # Формирование результата с метаданными
        result = {
            'probability': str(prediction_probability),
            'predicted_class': predicted_class,
            'metadata': {
                "audio_name": str(data).split('/')[-1],
                "duration": f"{duration:.1f} seconds",
                "format": format_info,
            }
        }
        return result