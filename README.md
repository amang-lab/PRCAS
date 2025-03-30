This is a project aiming to monitor pedestrians crossing road unsafely

Used dataset: coco2017

Trained the data for only the person class

detect and track people in provided videos

test.py : testing the accuarcy
train_data.py: train the model
train.py: train the YOLO model for only 'Person' class
init.py: extract data from training the model
lane_detection.py: canny edge detection
load_dataset: extract 'Person' class from COCO2017

---

**Run test code step:**

sudo apt install python3.12-venv

python3 -m venv pyvenv

source pyvenv/bin/activate

pip install -r requirements.txt

python3 test.py
deactivate
[by Amang's Research Group]
