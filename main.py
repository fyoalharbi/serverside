import subprocess
from uuid import uuid4
from flask import Flask, make_response, request, jsonify
from google.cloud import firestore
#import firebase_admin
#from firebase_admin import credentials
from flask import g 
import random
import wave
import librosa
from pydub import AudioSegment
#install /content/deep-speaker
#numpy matplotlib.pyplot firebase_admin uuid flask google.cloud deep_speaker.audio deep_speaker.batcher deep_speaker.constants deep_speaker.conv_models deep_speaker.test  
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
import tensorflow as tf
from keras.models import load_model
import numpy as np
import ffmpeg
from io import BytesIO
#from firebase import firebase
#from bs4 import BeautifulSoup #to download voices
#import os
#import random
#import numpy as np
#import matplotlib.pyplot as plt
#import subprocess

fileslist = []
UIDlist = []
file_id = "1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP"
output_file = "ResCNN_triplet_training_checkpoint_265.h5"


app = Flask(__name__)
#cred = credentials.Certificate('key.json')
#firebase_admin.initialize_app(cred)

#db = firestore.Client()
#recordings = db.collection('recordings')
#users = db.collection('users')
#x = recordings.document('EHw3AB1gXkevTYa8iwlXQUQkzxy2').get()

#model = DeepSpeakerModel()
np.random.seed(123)
random.seed(123)

# Define the model here.
model = DeepSpeakerModel()

model.m.load_weights('ResCNN_softmax_pre_training_checkpoint_102.h5', by_name=True)
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import onnx
"""
def convert_audio_bit_rate(audio_file_path, target_bit_rate=16000):
    output_file_path = "converted_audio.wav"
    try:
        # Delete the existing converted file if it exists
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        
        subprocess.run([
            "ffmpeg",
            "-i", audio_file_path,
            "-ab", str(target_bit_rate),
            output_file_path
        ])
    except Exception as e:
        print(f"Error in converting audio: {e}")
        return None
    return output_file_path
"""
@app.route('/')
def index():
        return "Hello"

@app.route('/greetings/<Uid>', methods=['GET', 'POST'])
def greetings(Uid):
     return "Hello, " + Uid

@app.route('/convert', methods=['POST'])
def convert_to_wav(audio_file):
    
    try:
    # Convert the file using FFmpeg securely
        subprocess.run(['ffmpeg', '-i', audio_file, '-codec:a', 'libmp3lame', '-q:a', '2', output_path], check=True)
    except subprocess.CalledProcessError:
        return jsonify(error="Conversion failed"), 500
        
        
        """
        aac_data = audio_file.read()
        aac_audio = AudioSegment.from_file(BytesIO(aac_data), format="aac")
        wav_data = aac_audio.export(format="wav").read() 
        return wav_data
    except Exception as e:
        return f'An error occurred: {str(e)}'
        """
        """
        out, err = (ffmpeg.
                    input(audio_file)
                    .output('pipe:' ))
        
        # Read the content of the audio file
        audio_content = audio_file.read()

        # Check if the file content is empty
        if len(audio_content) == 0:
            return "Audio content is empty"

        # Specify the ffmpeg command to convert the audio to WAV
        ffmpeg_command = ['ffmpeg',  '-i', '-', 'output.wav']

        # Start ffmpeg process
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Pass the audio content to ffmpeg process
        wav_output, error_output = process.communicate(input=audio_content)

        # Optionally, you can handle the output or errors from ffmpeg here

        return wav_output
"""
    except Exception as e:
        return f'An error occurred: {str(e)}', 500

@app.route('/train/<Uid>', methods=["POST", "GET"])
def train(Uid):
     
    
    try:        
        if 'audio' not in request.files:
            return 'No file provided', 400
        #gets file
        
        audio_file = request.files['audio']
        audio,sample_rate = librosa.load(audio_file, sr=16000)

        fileslist.append(audio)
        #add mfcc to it after manipulation
        UIDlist.append(Uid)
        
        mfcc = sample_from_mfcc(librosa.feature.mfcc(y=audio,sr= SAMPLE_RATE), NUM_FRAMES)
        mfcc = np.expand_dims(mfcc, axis=0)
        
        #mfcc.reshape(1, 160, 64, 1)
        mfcc = np.resize(mfcc, (1,160,64,1))
        #prediction = model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc(voice, SAMPLE_RATE)), axis=0))

        #prediction = model.m.predict()
        prediction = model.m.predict(np.expand_dims(mfcc,-1))#.reshape(1, 160, 64, 1))
        #model.m.predict(np.expand_dims(np.resize(sample_from_mfcc(librosa.feature.mfcc(y=audio,sr= SAMPLE_RATE), NUM_FRAMES), (1,160,64,1)),-1))
        # Compute the cosine similarity with each reference speaker
        similarities = [batch_cosine_similarity(prediction, model.m.predict((np.expand_dims(np.resize(sample_from_mfcc(librosa.feature.mfcc(y=audio,sr= SAMPLE_RATE), NUM_FRAMES), (1,160,64,1)),-1)))) for x in fileslist]
        predicted_speaker_index = np.argmax(similarities)
        predicted_speaker = UIDlist[predicted_speaker_index]

        
        return str(predicted_speaker)
    except Exception as e:
        return f"An Error occured: {e}"
def gcp_entry(request):
     return 'OK'
if __name__ == '__main__':
     app.run(host='10.0.2.2',debug=True,port=8000)        
