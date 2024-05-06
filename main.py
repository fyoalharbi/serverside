from uuid import uuid4
from flask import Flask, make_response, request, jsonify
from google.cloud import firestore
#import firebase_admin
#from firebase_admin import credentials
from flask import g 
import random
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
#from firebase import firebase
#from bs4 import BeautifulSoup #to download voices
#import os
#import random
#import numpy as np
#import matplotlib.pyplot as plt
#import subprocess


file_id = "1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP"
output_file = "ResCNN_triplet_training_checkpoint_265.h5"

# Execute the curl command to download the file
#curl_command = f"curl -L -o {output_file} 'https://drive.google.com/uc?export=download&id={file_id}'"
#subprocess.run(curl_command, shell=True, check=True)
# Set the random seeds for reproducibility
#np.random.seed(123)
#random.seed(123)
main = Flask(__name__)
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

# Load the checkpoint. https://drive.google.com/file/d/1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP.
# Also available here: https://share.weiyun.com/V2suEUVh (Chinese users).
model.m.load_weights('ResCNN_softmax_pre_training_checkpoint_102.h5', by_name=True)
#model = load_model('ResCNN_softmax_pre_training_checkpoint_102.h5')
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import onnx

@main.route('/')
def index():
        return "Hello"

@main.route('/greetings/')
def greetings():
     return "Hello"


@main.route('/train/<UID>', methods=["POST"])
def train(UID):
    
    try:
        if 'audio' not in request.files:
            return 'No file provided', 400
        #gets file
        file = request.files["audio"]
        files += file
        #gets UID
        UIDlist += UID
        #user = request.json['users']
        
             
        mfcc = sample_from_mfcc(read_mfcc(file, SAMPLE_RATE), NUM_FRAMES)
                # Predict the speaker using the model
        prediction = model.m.predict(np.expand_dims(mfcc, axis=0))
                # Compute the cosine similarity with each reference speaker
        similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc(file, SAMPLE_RATE), NUM_FRAMES), axis=0))) for ref_speaker in UIDlist]
        predicted_speaker_index = np.argmax(similarities)
        predicted_speaker = UIDlist[predicted_speaker_index]

        return jsonify({"predicted_speaker",predicted_speaker}), 200
    except Exception as e:
        return f"An Error occured: {e}"

def gcp_entry(request):
     return 'OK'
if name == '__main__':
     main.run(host='10.0.2.2',debug=True,port=8000)        
