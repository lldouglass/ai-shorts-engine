import os, requests, base64, re, wave
from pathlib import Path
from dotenv import load_dotenv
root = Path(r"C:\Users\Logan\Downloads\ai-shorts-engine")
load_dotenv(root / '.env')
key = os.getenv('GOOGLE_API_KEY')
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={key}"
text='You just got caught brain rotting. Now use that attention to learn one thing that saves you money.'
payload={
  'contents':[{'parts':[{'text':text}]}],
  'generationConfig':{
    'responseModalities':['AUDIO'],
    'speechConfig':{'voiceConfig':{'prebuiltVoiceConfig':{'voiceName':'Kore'}}}
  }
}
r=requests.post(url,json=payload,timeout=60)
print('status',r.status_code)
r.raise_for_status()
part=r.json()['candidates'][0]['content']['parts'][0]['inlineData']
raw=base64.b64decode(part['data'])
mime=part.get('mimeType','')
rate=24000
m=re.search(r'rate=(\d+)',mime)
if m: rate=int(m.group(1))
out=root/'output'/'samples'/'gemini_tts_test.wav'
out.parent.mkdir(parents=True,exist_ok=True)
with wave.open(str(out),'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    wf.writeframes(raw)
print('saved',out,'bytes',out.stat().st_size,'rate',rate)
