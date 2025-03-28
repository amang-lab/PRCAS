# Import the YOLO module from Ultralytics
import os
from multiprocessing import freeze_support, Process

from ultralytics import YOLO
import torch

"""
This is the model to train the filtered COCO dataset from load_dataset.py
"""
def train(model):
    # model.to('cuda')

    results = model.train(
        data='/trained_model/dataset.yaml', epochs=10, imgsz=640,batch=-1,
        name='yolov8n_people_50e',device=0
    )
    model.save("yolov8n_person.pt")
    print(results)


if __name__=='__main__':
    print(os.path.isfile("/trained_model/dataset.yaml"))
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")
    train(model)
    #p=Process(target=train, args=(model,))
    #p.start()
    #p.join()



