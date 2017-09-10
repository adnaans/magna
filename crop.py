from google.cloud import vision
from google.cloud.vision import types
from google.cloud import translate
from PIL import Image, ImageDraw, ImageFont
import io
import textwrap
import json
import base64
import requests

client = vision.ImageAnnotatorClient()
translate_client = translate.Client()
target = 'en'
font_path = 'animeace.ttf'
font = ImageFont.truetype(font_path, 8, encoding='unic')

def translate(b64):
    # with io.open("a.png", 'rb') as image_file:
    #     content = image_file.read()

    # image = types.Image(content=b64)
    im = Image.open(io.BytesIO(b64))
    im.save('temp2.png', 'png')
    r = requests.get("http://localhost:8000/pic")
    bounds = r.json()
    print(bounds)
    draw = ImageDraw.Draw(im)

    # bounds = [([300,10],[400,200]),([450,500],[560,700]),([303,300],[376,472]),([230,345],[296,450]),([60,267],[120,380])]
    c = 0
    for bound in bounds:
        c += 1
        im2 = im.crop((bound[0], bound[1], bound[2], bound[3]))
        buffer = io.BytesIO()
        im2.save(buffer, 'png')
        im2.save("k.png", "png")
        # image = types.Image(content=buffer.getvalue())
        # response = client.text_detection(image=image)
        r = requests.post("https://vision.googleapis.com/v1/images:annotate?key=AIzaSyCkQTi_QOKR2L6UQiRaxvkAuz1VEf4yX0I", data = json.dumps({"requests": [{"image": {"content": base64.b64encode(buffer.getvalue())},"features":[{"type": "TEXT_DETECTION"}]}]}))
        response = r.json()
        if 'textAnnotations' in response['responses'][0]:
            print(response['responses'])
            texts = response['responses'][0]['textAnnotations']
        print('Texts:')

        for text in texts[:1]:
            print('"{}"'.format(text['description'].encode('utf-8')))
            r = requests.get("https://translation.googleapis.com/language/translate/v2?target=en&source=ja&q={}&key={}".format(text['description'].encode('utf-8'),'AIzaSyCkQTi_QOKR2L6UQiRaxvkAuz1VEf4yX0I'))
            translation = json.loads(r.content)
            print(translation)
            # translation = translate_client.translate(
            #     text.description.encode('utf-8'),
            #     target_language=target)
            translatedText=translation['data']['translations'][0]['translatedText']
            print(text['boundingPoly']['vertices'])
            vertices = ([(vertex['x'], vertex['y']) for vertex in text['boundingPoly']['vertices']])
            # print(vertices[0][0]+bound[0][0], vertices[0][1]+bound[0][1],vertices[0][0]+bound[0][0]+abs(vertices[0][0]-vertices[2][0]), vertices[0][1]+bound[0][1]+abs(vertices[0][1]-vertices[2][1]))
            draw.rectangle([
                vertices[0][0]+bound[0]-5, vertices[0][1]+bound[1]-5,
                vertices[0][0]+bound[0]+abs(vertices[0][0]-vertices[2][0])+5, vertices[0][1]+bound[1]+abs(vertices[0][1]-vertices[2][1])+5], 'white', None)
            print('bounds: {}'.format(vertices))
            maxwidth = abs(vertices[0][0] - vertices[2][0])+10;
            maxheight = abs(vertices[0][1] - vertices[2][1])+10;
            width, height = font.getsize(translatedText)
            y_text = height
            print(maxwidth/float(width)*len(translatedText))

            lines = textwrap.wrap(translatedText, width=int(maxwidth/float(width)*len(translatedText))+1)
            for line in lines:
                print(line)
                draw.text((vertices[0][0]+bound[0]-5, vertices[1][1] + bound[1]+ y_text), line.replace("&#39;","'"), font=font, fill="Black")
                y_text += height+maxheight/len(lines)/2

    buffer = io.BytesIO()
    im.save(buffer, 'png')
    return buffer.getvalue()
    # im.save('output-hint.jpg', 'Png')
