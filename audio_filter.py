from vosk import Model, KaldiRecognizer, SetLogLevel
import moviepy.editor as mp
import json
import subprocess
import os
from pydub import AudioSegment

ffmpeg_ = os.getcwd()+ "\\ffmpeg.exe" 

AudioSegment.converter = os.getcwd()+ "\\ffmpeg.exe"                    
AudioSegment.ffprobe   = os.getcwd()+ "\\ffprobe.exe"

sample_rate=16000
model = Model("vosk-model-small-ru-0.22") # полный путь к модели

rec = KaldiRecognizer(model, sample_rate)
rec.SetWords(True)

blacklist = open("BL.txt", "r").read()
blacklist = blacklist.split("\n")

def transcribe(input_file, output_pref):
    mute_time= []
    results = []
    content = ''
    process = subprocess.Popen([ffmpeg_, '-loglevel', 'quiet', '-i',
                                input_file,
                                '-ar', str(sample_rate) , '-ac', '1', '-f', 's16le', '-'],
                                stdout=subprocess.PIPE)

    while True:
       data =  process.stdout.read(2000)
       if len(data) == 0:
           break
       if rec.AcceptWaveform(data):
           results.append(rec.Result())
    results.append(rec.FinalResult())
    for i, res in enumerate(results):
        words = json.loads(res).get('result')
        if not words:
            continue
        for w in words:
            if w['word'] in blacklist:
                mute_time.append({"time_start": w['start'], "time_end": w['end']})
        #content += ' '.join([w['word'] for w in words])
    #print(content)
    audioseg = AudioSegment.from_file(os.getcwd()+"\\"+ input_file, "mp4")
    censoring = AudioSegment.from_mp3(os.getcwd()+"\\beep.mp3")
    for entry in mute_time:
        begin = entry["time_start"] * 1000
        end = entry["time_end"] * 1000
        fpart = audioseg[:begin] + censoring[:end - begin]
        lpart= audioseg[end:]
        audioseg = fpart + lpart

    json_result = {"result": mute_time}

    with open(str(output_pref + '_audio.json'), 'w') as f:
        json.dump(json_result, f)

    audioseg.export(os.getcwd()+"\\result.mp3", format="mp3")

    os.system(str("ffmpeg -i " + input_file + " -i result.mp3 -map 0:0 -map 1:0 -c:v copy -c:a aac -b:a 256k -shortest " + output_pref
    +  "_result.mp4"))
    os.remove("result.mp3")

#transcribe("hackathon_part_1.mp4", "output")