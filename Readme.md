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

We provide scripts that can be used in most of the tracking by detection algorithms listed in the paper. For ByteSSM, we provide the evaluation script and the configuration file for training. For Co-MOT and MeMOTR, we provide the configuration files for training. 

1. Follow the installation instructions of your desired tracking by detection algorithm.
   
2. Install SAHI and Ultralytics
~~~
pip install -U ultralytics sahi
~~~
3. Download trained models [here](https://1drv.ms/f/s!App_ySGnU8ijvP5uIw1qva19CuLv_w?e=UPT23N). Put in folder named **pretrained**.

4. Move respective script into **tools** folder or **root** if tools does not exist. 

5. For example, run:
~~~
python tools/bot_sort_demo_yolov11.py
~~~
6. Simply run the evaluation code:
```
python scripts/run_mot_challenge.py --GT_FOLDER slope_track --BENCHMARK slope_track --METRICS HOTA CLEAR Identity --TRACKERS_FOLDER yolo11/slopetrack --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```

### DeepEIoU + GlideTrack

1. Follow the installation instructions of [DeepEIoU](https://github.com/hsiangwei0903/Deep-EIoU).
   
2. Install SAHI and Ultralytics
~~~
pip install -U ultralytics sahi
~~~

3. Install Mamba
~~~
pip install mamba-ssm[causal-conv1d]
~~~

4. Download trained models [here](https://1drv.ms/f/s!App_ySGnU8ijvP5uIw1qva19CuLv_w?e=UPT23N). Put in folder named **pretrained**.

5. Move the scripts in [tracker](https://github.com/SlopeTrack/Slope_Track/tree/main/scripts/DeepEIoU_GlideTrack/tracker) into **tracker** folder and this scripts in [tools](https://github.com/SlopeTrack/Slope_Track/tree/main/scripts/DeepEIoU_GlideTrack/tools) into **tools** folder.

6. Run 
~~~
python tools/deep_eiou_yolov11_glidetrack.py --glide_weights NAME_OF_MODEL --glide_label NAME_OF_SAVE_FILE --in_dim 36 --num_freqs 8 --split test
~~~

## License
The code is under the Modified BSD License.

## Acknowledgement  
The evaluation metrics and code are from [MOT Challenge](https://motchallenge.net/) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). The analysis code is based on [DanceTrack](https://github.com/DanceTrack/DanceTrack) and [SportsMoT](https://github.com/MCG-NJU/SportsMOT). The ReID framework is from [FastReid](https://github.com/JDAI-CV/fast-reid). Thank you for your amazing work!

**This research work is being carried out as part of a collaborative i-Démo Regionalized project under the French government's regionalized France 2030 program. It was financed via Bpifrance by the French government, the Auvergne-Rhône-Alpes Region and Grenoble Alpes Métropole.**

~~~
@article{Campbell2026SlopeTrack,
          title   = {Slope-Track: Multiple Object Tracking on Ski Slopes},
          author  = {Campbell, M'Saydez and Ducottet, Christophe and Muselet, Damien and Emonet, R{\'e}mi},
          journal = {Computer Vision and Image Understanding},
          pages   = {104663},
          year    = {2026},
          issn    = {1077-3142},
          doi     = {10.1016/j.cviu.2026.104663},
          url     = {https://www.sciencedirect.com/science/article/pii/S1077314226000305}
        }
        
~~~





