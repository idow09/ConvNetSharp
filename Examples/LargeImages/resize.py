import os
from PIL import Image

for d in ['Train', 'Test']:
    for f in os.listdir(d):
        try:
            image = Image.open(d + '/' + f)
            image.thumbnail((400, 400))
            image.save('../MediumImages/' + d + '/' + f.replace('Large', 'Medium'))
        except Exception as e:
            print(e)

