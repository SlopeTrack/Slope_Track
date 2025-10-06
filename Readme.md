# Slope-Track


## News
- (05/2025) Uploaded code and instructions for downloading and evaluating the dataset.
- (10/2025) Updated  code for evaluating the dataset and uploaded code for training the motion model.

## Dataset Organization
~~~
{Slope_Track ROOT}
|-- slope_track
|   |-- train
|   |   |-- slope_track000001
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- val
|   |   |-- ...
|   |-- test
|   |   |-- ...
|-- unlabled_test
|   |   |-- slope_track0000021
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- det
|   |   |   |   |-- det.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- train_seqmap.txt
|   |-- val_seqmap.txt
|   |-- test_seqmap.txt
|   |-- unlabeled_test_seqmap.txt
~~~
Annotations are aligned as follows: 
~~~
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
~~~

## Training

### Detection

For detection, we utilized the [YOLOv11](https://docs.ultralytics.com/models/yolo11/) large model from [Ultralytics](https://github.com/ultralytics/ultralytics). For details, please see [training instructions](detection_training/Readme.md). 

### Re-identification

For reID, we followed the instructions from [HybridSORT](https://github.com/ymzis69/HybridSORT). 

Generate patches:
~~~
python fast_reid/datasets/generate_slopetrack_patches.py --data_path slope_track
~~~
Train the reID model:
~~~
python fast_reid/tools/train_net.py --config-file ./fast_reid/configs/SlopeTrack/sbs_S50.yml MODEL.DEVICE "cuda:0"
~~~

You can see [HybridSORT](https://github.com/ymzis69/HybridSORT) and [FastReid](https://github.com/JDAI-CV/fast-reid) pages for more details.

## Motion Modelling

For motion, we utilizes the [simple Mamba block](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) from [Mamba](https://github.com/state-spaces/mamba). For details, please see [training instructions](motion_training/Readme.md).

## Evaluation

We provide scripts that can be used in the tracking algorithms listed in the paper.

1. Follow the installation instructions of your desired tracking algorithm.
   
2. Install SAHI and Ultralytics
~~~
pip install -U ultralytics sahi
~~~
3. Download trained models [here](https://1drv.ms/f/s!App_ySGnU8ijvP5uIw1qva19CuLv_w?e=UPT23N). Put in folder named **pretrained**.

4. Move respective script into **tools** folder or **root** if tools does not exist. 

5. For example, run:
~~~
python tools/bot_yolov11.py
~~~
6. Simply run the evaluation code:
```
python scripts/run_mot_challenge.py --GT_FOLDER slope_track --BENCHMARK slope_track --METRICS HOTA CLEAR Identity --TRACKERS_FOLDER yolo11/slopetrack --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```


## License
The code is under the Modified BSD License.

## Acknowledgement  
The evaluation metrics and code are from [MOT Challenge](https://motchallenge.net/) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). The analysis code is based on [DanceTrack](https://github.com/DanceTrack/DanceTrack) and [SportsMoT](https://github.com/MCG-NJU/SportsMOT). The ReID framework is from [FastReid](https://github.com/JDAI-CV/fast-reid). Thank you for your amazing work!

**This research work is being carried out as part of a collaborative i-Démo Regionalized project under the French government's regionalized France 2030 program. It was financed via Bpifrance by the French government, the Auvergne-Rhône-Alpes Region and Grenoble Alpes Métropole.**



