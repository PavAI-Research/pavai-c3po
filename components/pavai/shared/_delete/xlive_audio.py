
### https://www.codecnetworks.com/blog/playing-and-saving-audio-in-python/


### Simple Audio
# import simpleaudio as sa 
# filename = 'myfile.wav'
# wave_obj = sa.WaveObject.from_wave_file(filename)
# play_obj = wave_obj.play()
# play_obj.wait_done()


### Win Sound use Windows:

# import winsound 
# filename = 'myfile.wav'
# winsound.Playsound(filename,winsound.SND_FILENAME)
# winsound.Beep(1000,100)

### Python-sound device

import sounddevice as sd
import soundfile as sf

filename='myfile.wav'
# extract data and sampling rate from file
data, fs = sf.read(filename,dtype='float32')
sd.play(data,fs)
status = sd.wait() # wait until file is done playing

## PYDUB
from pydub import AudioSegment
from pydub.playback import play
sound = AudioSegment.from_wav('myfile.wav')
play(sound)

# sound = AudioSegment.from_file('myfile.mp3', format='mp3')
# sound.export('myfile.wav',format='wav')


## pip install ffmpeg-python


