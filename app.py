from uuid import uuid4
from flask import Flask, make_response, request, jsonify
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
#install /content/deep-speaker

from deep_speaker.audio import read_mfcc
from deep_speaker.batcher import sample_from_mfcc
from deep_speaker.constants import SAMPLE_RATE, NUM_FRAMES
from deep_speaker.conv_models import DeepSpeakerModel
from deep_speaker.test import batch_cosine_similarity
app = Flask(__name__)
cred = credentials.Certificate('key.json')
db = firestore.Client()
recordings = db.collection('recordings')
#users= db.collection('users').document('0jRPyBKpcIXuugXQl2As7WA5xP93')
#res = users.collection('0jRPyBKpcIXuugXQl2As7WA5xP93')


import os
import random
import numpy as np
import matplotlib.pyplot as plt

model = DeepSpeakerModel()



@app.route('/')
def index():
    return "Hello"

@app.route('/train', methods=["GET"])
def train():
    try:
        file = request.files["file"]
        user = request.json['users']
        
        mfcc = sample_from_mfcc(read_mfcc(file, SAMPLE_RATE), NUM_FRAMES)
        
                # Predict the speaker using the model
        prediction = model.m.predict(np.expand_dims(mfcc, axis=0))

                # Compute the cosine similarity with each reference speaker
        similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(file)))]

        predicted_speaker_index = np.argmax(similarities)
        predicted_speaker = [predicted_speaker_index]

        return jsonify({"success": True}), 200
    except Exception as e:
        return f"An Error occured: {e}"
    
        

@app.route('/verify', methods=['POST', 'GET'])
def verify():
    try:
        
        file = request.files["file"]
        voice_ref = db.collection('recordings')
        user = request.json['users']
        voice = voice_ref.document(user)
        """
        mfcc = sample_from_mfcc(read_mfcc(voice, SAMPLE_RATE), NUM_FRAMES)
        
                # Predict the speaker using the model
        prediction = model.m.predict(np.expand_dims(mfcc, axis=0))

                # Compute the cosine similarity with each reference speaker
        similarities = [batch_cosine_similarity(prediction, model.m.predict(np.expand_dims(voice)))]

        predicted_speaker_index = np.argmax(similarities)

                # Get the predicted speaker
        predicted_speaker = [predicted_speaker_index]
        """
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

    
"""

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