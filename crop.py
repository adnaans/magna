from google.cloud import vision
from google.cloud.vision import types
from google.cloud import translate
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap
import json

client = vision.ImageAnnotatorClient()
translate_client = translate.Client()
target = 'en'
font_path = 'animeace.ttf'
font = ImageFont.truetype(font_path, 16, encoding='unic')

with io.open("a.png", 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
im = Image.open(io.BytesIO(content))
draw = ImageDraw.Draw(im)

response = client.text_detection(image=image)
texts = response.text_annotations
print('Texts:')

for text in texts[:1]:
    print('\n"{}"'.format(text.description.encode('utf-8')))
    translation = translate_client.translate(
        text.description.encode('utf-8'),
        target_language=target)
    translatedText=translation['translatedText']

    vertices = ([(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices])
    draw.rectangle([
        vertices[1][0]+10, vertices[1][1],
        vertices[3][0], vertices[3][1]+10], 'white', None)
    print('bounds: {}'.format(vertices))
    maxwidth = abs(vertices[1][0] - vertices[3][0])+10;
    maxheight = abs(vertices[1][1] - vertices[3][1])+10;
    width, height = font.getsize(translatedText)
    y_text = height
    print(maxwidth/float(width)*len(translatedText))

    lines = textwrap.wrap(translatedText, width=int(maxwidth/float(width)*len(translatedText))+1)
    for line in lines:
        print(line)
        draw.text((vertices[0][0]-10, vertices[1][1] + y_text), line, font=font, fill="Black")
        y_text += height+maxheight/len(lines)/2


im.save('output-hint.jpg', 'Png')
