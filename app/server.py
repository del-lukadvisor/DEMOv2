from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional

import cv2
import imutils
import numpy as np

import torch
import base64
import random

app = FastAPI()
templates = Jinja2Templates(directory = 'templates')

colors = [tuple([random.randint(0, 255) for _ in range(3)]) for _ in range(100)] #for bbox plotting

##############################################
#-------------GET Request Routes--------------
##############################################
@app.get("/")
def home(request: Request):
    ''' Returns html jinja2 template render for home page form
    '''

    return templates.TemplateResponse('home.html', {
            "request": request,
        })

##############################################
#------------POST Request Routes--------------
##############################################
@app.post("/")
def detect_with_server_side_rendering(request: Request,
                        file_list: List[UploadFile] = File(...)):

    model = torch.hub.load("WongKinYiu/yolov7", "custom", "./best.pt", source="github", force_reload=True)
    model.conf = 0.5
    model.agnostic = False

    img_batch = [cv2.imdecode(np.fromstring(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
                    for file in file_list]

    #create a copy that corrects for cv2.imdecode generating BGR images instead of RGB
    #using cvtColor instead of [...,::-1] to keep array contiguous in RAM
    img_batch_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_batch]

    results = model(img_batch_rgb, size=640)
    
    #Check number of Pop vs Products
    pop_products = results.pandas().xyxy[0].sort_values('ymax')
    
    pop = 0
    products = 0
    for i in pop_products['name']:
        if i == 'pop':
             pop += 1
        else:
             products += 1

    #Check Position of Products vs. Pop -- Example: Pop at the beginning
    pop_products_array = pop_products.to_numpy()
    row = 0
    position = 0
    y = 0
    for i in pop_products_array:
        if i[6] == 'pop':
        #Grab xmin value
            xmin = i[0]
            row += 1
            for x in pop_products_array:
                if x[0] != xmin: 
                #Only perform if ymin is within +- 25 
                    if x[3] <= i[3] + 30 and x[3] >= i[3] - 30:
                         y = 0
                         if xmin > x[0]:
                             y += 1
                             position = row

    json_results = results_to_json(results, model)

    img_str_list = []
    #plot bboxes on the image
    for img, bbox_list in zip(img_batch, json_results):
        for bbox in bbox_list:
            label = f'{bbox["class_name"]} {bbox["confidence"]:.2f}'
            plot_one_box(bbox['bbox'], img, label=label, 
                    color=colors[int(bbox['class'])], line_thickness=3)

        img_str_list.append(base64EncodeImage(img))

    #escape the apostrophes in the json string representation
    encoded_json_results = str(json_results).replace("'",r"\'").replace('"',r'\"')

    return templates.TemplateResponse('show_results.html', {
            'request': request,
            'img_str_list': img_str_list, #unzipped in jinja2 template
            'bbox_data_str': encoded_json_results,
            'pop': pop,
            'products': products,
            'y': y,
            'position': position
        })

##############################################
#--------------Helper Functions---------------
##############################################

def results_to_json(results, model):
    ''' Converts yolo model output to json (list of list of dicts)'''
    return [
                [
                    {
                    "class": int(pred[5]),
                    "class_name": model.model.names[int(pred[5])],
                    "bbox": [int(x) for x in pred[:4].tolist()], #convert bbox results to int from float
                    "confidence": float(pred[4]),
                    }
                for pred in result
                ]
            for result in results.xyxy
            ]


def plot_one_box(x, im, color=(128, 128, 128), label=None, line_thickness=3):
    # Directly copied from: https://github.com/ultralytics/yolov5/blob/cd540d8625bba8a05329ede3522046ee53eb349d/utils/plots.py
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def base64EncodeImage(img):
    ''' Takes an input image and returns a base64 encoded string representation of that image (jpg format)'''
    _, im_arr = cv2.imencode('.jpg', img)
    im_b64 = base64.b64encode(im_arr.tobytes()).decode('utf-8')
    return im_b64

if __name__ == '__main__':
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default = 'localhost')
    parser.add_argument('--port', default = 8000)
    opt = parser.parse_args()
    
    app_str = 'server:app' #make the app string equal to whatever the name of this file is
    uvicorn.run(app_str, host= opt.host, port=opt.port, reload=True)
