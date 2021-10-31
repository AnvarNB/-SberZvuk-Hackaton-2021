from flask import Flask, request, json
import urllib 
import audio_filter
import os

app = Flask(__name__)


@app.route('/', methods=['POST'])
def show_user_profile():
    source = request.json['source']
    print(source)
    prefix = request.json["prefix"]
    print(prefix)
    urllib.request.urlretrieve(source, "test.mp4")
    audio_filter.transcribe("test.mp4", prefix)
    os.remove("test.mp4")
    response = app.response_class(
        response=json.dumps({"code":"200","message":"OK"}),
        status=200)
    return response


@app.route('/face_detect', methods=['POST'])
def video_profile():
    source = request.json['source']
    print(source)
    prefix = request.json["prefix"]
    print(prefix)
    urllib.request.urlretrieve(source, "test.mp4")

    audio_filter.transcribe("test.mp4", prefix)
    os.remove("test.mp4")

    response = app.response_class(
        response=json.dumps({"code": "200", "message": "OK"}),
        status=200)
    return response
