# Training the Detection Model 

Install ultralytics and SAHI
~~~
pip install -U ultralytics sahi
~~~

1. Split the frames into 640 x 640 with a 0.1 overlap using SAHI.
2. Convert to YOLO format.

~~~
python slice_images.py
~~~

Run
~~~
python training.py 
~~~

We used 3 gpus to train the model
