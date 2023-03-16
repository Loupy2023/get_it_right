import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder


url = "http://127.0.0.1:8000/upload_image"

filename = "image_loup.jpg"
m = MultipartEncoder(
        fields={'file': ('filename', open(filename, 'rb'), 'image/jpeg')}
    )
r = requests.post(url, data=m, headers={'Content-Type': m.content_type}, timeout = 20_000)
assert r.status_code == 20
