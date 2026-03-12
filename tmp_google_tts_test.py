import os, json, base64, requests
from pathlib import Path
from dotenv import load_dotenv

root = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine")
load_dotenv(root / '.env')
key = os.getenv('GOOGLE_API_KEY')
print('has_key', bool(key))
if not key:
    raise SystemExit(1)
url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={key}"
payload = {
  "input": {"text": "You just got caught brain rotting. Now learn one thing that saves you money."},
  "voice": {"languageCode": "en-US", "name": "en-US-Neural2-D"},
  "audioConfig": {"audioEncoding": "MP3", "speakingRate": 1.02, "pitch": 0.0}
}
r = requests.post(url, json=payload, timeout=30)
print('status', r.status_code)
print('body', r.text[:400])
if r.ok:
    data = r.json()
    audio_b64 = data.get('audioContent')
    if audio_b64:
        out = root / 'output' / 'samples' / 'google_tts_test.mp3'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(audio_b64))
        print('saved', out)
