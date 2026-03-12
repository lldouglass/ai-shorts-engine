import requests
html=requests.get('https://mixkit.co/free-stock-music/',headers={'User-Agent':'Mozilla/5.0'},timeout=20).text
needle='data-audio-player-item-id-value="127"'
idx=html.find(needle)
print('idx',idx)
if idx!=-1:
    s=max(0,idx-2000)
    e=min(len(html),idx+2500)
    print(html[s:e])
