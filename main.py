from flask import Flask, make_response, request, jsonify
import random
import wave
import librosa
from pydub import AudioSegment
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
import tensorflow as tf
from keras.models import load_model
import numpy as np
import os
from multiprocessing import Value, Lock

app = Flask(__name__)

# Initialize shared variables and locks
fileslist = []
fileslist_lock = Lock()

UIDlist = []
UIDlist_lock = Lock()

np.random.seed(123)
random.seed(123)

# Load the DeepSpeakerModel
model = DeepSpeakerModel()
model.m.load_weights('ResCNN_softmax_pre_training_checkpoint_102.h5', by_name=True)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

@app.route('/train/<Uid>', methods=["POST", "GET"])
def train(Uid):
    try:
        if 'audio' not in request.files:
            return 'No file provided', 400

        audio_file = request.files['audio']
        audio, sample_rate = librosa.load(audio_file, sr=16000)

        with fileslist_lock:
            fileslist.append(audio)

        with UIDlist_lock:
            UIDlist.append(Uid)
        
        mfcc = sample_from_mfcc(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE), NUM_FRAMES)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.resize(mfcc, (1, 160, 64, 1))
        prediction = model.m.predict(np.expand_dims(mfcc, -1))
        with fileslist_lock:
            similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(np.resize(sample_from_mfcc(librosa.feature.mfcc(y=x, sr=SAMPLE_RATE), NUM_FRAMES), (1, 160, 64, 1)),-1))) for x in fileslist]
            predicted_speaker_index = np.argmax(similarities)
            predicted_speaker = UIDlist[predicted_speaker_index]

        return str(predicted_speaker)
    except Exception as e:
        return f"An Error occurred: {e}"

@app.route('/auth', methods=["POST", "GET"])
def auth():
    try:
        if 'audio' not in request.files:
            return 'No file provided', 400

        audio_file = request.files['audio']
        audio, sample_rate = librosa.load(audio_file, sr=16000)

        mfcc = sample_from_mfcc(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE), NUM_FRAMES)
        mfcc = np.expand_dims(mfcc, axis=0)
        mfcc = np.resize(mfcc, (1, 160, 64, 1))

        prediction = model.m.predict(np.expand_dims(mfcc, -1))
        with fileslist_lock:
            similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(np.resize(sample_from_mfcc(librosa.feature.mfcc(y=x, sr=SAMPLE_RATE), NUM_FRAMES), (1, 160, 64, 1)),-1))) for x in fileslist]
            predicted_speaker_index = np.argmax(similarities)
            predicted_speaker = UIDlist[predicted_speaker_index]
            #return str(similarities)
        return str(predicted_speaker)
    except Exception as e:
        return f"An Error occurred: {e}"
    
if __name__ == 'main':
     app.run(host='10.0.2.2', debug=True, port=8000)
