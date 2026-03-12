import os, requests, base64
from pathlib import Path
from dotenv import load_dotenv
root = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine")
load_dotenv(root / '.env')
key = os.getenv('GOOGLE_API_KEY')
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={key}"
payload = {
  "contents": [{"parts": [{"text": "You just got caught brain rotting. Now let's learn one thing that saves you money."}]}],
  "generationConfig": {
    "responseModalities": ["AUDIO"],
    "speechConfig": {
      "voiceConfig": {
        "prebuiltVoiceConfig": {"voiceName": "Kore"}
      }
    }
  }
}
r = requests.post(url, json=payload, timeout=60)
print('status', r.status_code)
print(r.text[:700])
if r.ok:
  data = r.json()
  parts = data.get('candidates',[{}])[0].get('content',{}).get('parts',[])
  if parts and 'inlineData' in parts[0]:
    b64 = parts[0]['inlineData']['data']
    mime = parts[0]['inlineData'].get('mimeType')
    out = root / 'output' / 'samples' / 'gemini_tts_test.raw'
    out.write_bytes(base64.b64decode(b64))
    print('saved', out, 'mime', mime, 'bytes', out.stat().st_size)
