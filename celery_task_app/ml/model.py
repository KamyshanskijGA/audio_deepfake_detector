import joblib
from keras.models import load_model
import numpy as np
import logging
import os
import librosa

# MODEL_PATH = os.environ['MODEL_PATH']
MODEL_PATH = rf'/mnt/c/Users/390/Desktop/2app/celery_task_app/ml/model_fin.h5'

class DetectionModel:

    """ Wrapper for loading and serving pre-trained model"""

    def __init__(self):
        self.model = self._load_model_from_path(MODEL_PATH)

    @staticmethod
    def _load_model_from_path(path):
        #model = joblib.load(path)
        model = load_model(MODEL_PATH)
        return model

    def extract_features(self, audio_file):
        features = []
        try:
            y, sr = librosa.load(audio_file)

            mfccs = librosa.feature.mfcc(y=y, sr=sr)
            chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)

            original_features = np.concatenate((np.mean(mfccs, axis=1), np.mean(chroma_cqt, axis=1), np.mean(tonnetz, axis=1)))
            logging.info(original_features)
            features.append(original_features)

        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
        features = np.expand_dims(features, axis=2)
        logging.info("_________________________________")
        logging.info(features)
        return features
        

    def predict(self, data, return_option='Prob'):
        """
        Make batch prediction on list of preprocessed feature dicts.
        Returns class probabilities if 'return_options' is 'Prob', otherwise returns class membership predictions
        """

        logging.info(f"test2: {data}")
        features = self.extract_features(data)
        prediction = self.model.predict(features)

        predicted_class = 'Fake' if prediction[0][0] < 0.5 else 'Human'
        # return (prediction[0][0], predicted_class)
        return prediction[0][0]



