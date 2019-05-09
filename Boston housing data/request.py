import requests

url = 'http://localhost:5000/api'

r = requests.post(url, json={'lstat':4.03, 'rm':7.185})
print(r.json())