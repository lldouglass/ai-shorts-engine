import requests,re
html=requests.get('https://mixkit.co/free-stock-music/',headers={'User-Agent':'Mozilla/5.0'},timeout=20).text
needle='https://assets.mixkit.co/music/127/127.mp3'
idx=html.find(needle)
print('idx',idx)
if idx!=-1:
    s=max(0,idx-700)
    e=min(len(html),idx+700)
    print(html[s:e])
