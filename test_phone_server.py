import os, urllib.request, urllib.error, json
import numpy as np
from phone_server import PhoneServer
from main import detect_pitch
os.environ['VOICE_PONG_PORT']='0'
s=PhoneServer()
try:
    def request(url,data=None):
        return urllib.request.urlopen(urllib.request.Request(url,data=data),timeout=3)
    assert all(url.startswith('http://') for url in s.urls)
    assert len(s.qrs)==2 and s.tokens[0]!=s.tokens[1]
    assert 'Player 1' in request(s.urls[0]).read().decode()
    for i in range(2):
        audio=(np.sin(np.arange(4096)*2*np.pi*(220+i*110)/48000)*.2).astype('<f4')
        assert request(s.urls[i].replace('/?','/audio?')+'&rate=48000',audio.tobytes()).status==200
        pitch,_=detect_pitch(s.latest[i][0])
        assert abs(pitch-(220+i*110))<3,pitch
        assert s.connected(i)
    assert json.loads(request(s.urls[0].replace('/?','/status?')).read())['paused']
    try:
        request(s.urls[0].replace(s.tokens[0],'invalid'))
        raise AssertionError('Invalid token accepted')
    except urllib.error.HTTPError as e:
        assert e.code==403
    print('PASS: HTTP, player links, isolated audio, resampling/pitch, status, token rejection')
finally:
    s.close()
