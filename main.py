from uuid import uuid4
from flask import Flask, make_response, request, jsonify
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
from flask import g 
#install /content/deep-speaker
#numpy matplotlib.pyplot firebase_admin uuid flask google.cloud deep_speaker.audio deep_speaker.batcher deep_speaker.constants deep_speaker.conv_models deep_speaker.test  
from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
from firebase import firebase
from bs4 import BeautifulSoup #to download voices
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import subprocess

file_id = "1F9NvdrarWZNktdX9KlRYWWHDwRkip_aP"
output_file = "ResCNN_triplet_training_checkpoint_265.h5"

# Execute the curl command to download the file
curl_command = f"curl -L -o {output_file} 'https://drive.google.com/uc?export=download&id={file_id}'"
subprocess.run(curl_command, shell=True, check=True)
# Set the random seeds for reproducibility
np.random.seed(123)
random.seed(123)
app = Flask(__name__)
cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred)

#db = firestore.Client()
#recordings = db.collection('recordings')
#users = db.collection('users')
#x = recordings.document('EHw3AB1gXkevTYa8iwlXQUQkzxy2').get()

model = DeepSpeakerModel()

@app.route('/')
def index():
        return "Hello"

@app.route('/train/<UID>', methods=["GET"])
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

        return jsonify({predicted_speaker: True}), 200
    except Exception as e:
        return f"An Error occured: {e}"

def gcp_entry(request):
     return 'OK'
        
""""
@app.route('/verify', methods=['POST', 'GET'])
def verify():
    try: 
        file = request.files["file"]
        voice_ref = db.collection('features')
        user = request.json['users']
        voice = voice_ref.document(user)
        
        mfcc = sample_from_mfcc(read_mfcc(voice, SAMPLE_RATE), NUM_FRAMES)
        
                # Predict the speaker using the model
        prediction = model.m.predict(np.expand_dims(mfcc, axis=0))

                # Compute the cosine similarity with each reference speaker
        similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(voice)))]

        predicted_speaker_index = np.argmax(similarities)

                # Get the predicted speaker
        predicted_speaker = [predicted_speaker_index]
        
        prediction = model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc(voice, SAMPLE_RATE)), axis=0))

                # Compute the cosine similarity with reference speakers trained on 'speakerX1' and 'speakerX2'
        similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(sample_from_mfcc(read_mfcc(voice, SAMPLE_RATE), NUM_FRAMES), axis=0)))]

                # Find the index of the reference speaker with the highest similarity
        predicted_speaker_index = np.argmax(similarities)

                # Get the predicted speaker
        predicted_speaker = [predicted_speaker_index]

        # Print the predicted speaker
        print('Actual Speaker:', user)
        print('Predicted Speaker:', predicted_speaker)


        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error occured: {e}"






if __name__ == '__main__':
    app.run(debug=True, threaded=True)
##@firestore.transactional
#def get_session_data(transaction, session_id):

    


@app.route('/train', method=['POST'])
def train():
    if request.method == 'POST':
        if 'voice' not in request.voice:
            return 'No voice found'
   




def identifyVoice(voice_path):
    voice = 

@app.route('/list', methods=['GET'])
def read():




@app.route('/delete', methods['GET', 'DELETE'])
def delete():

"""
