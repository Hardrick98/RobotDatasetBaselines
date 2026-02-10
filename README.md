# RobotDatasetBaselines

Place the dataset in the exo_dataset folder. Folder should look like this:

```
exo_dataset/
├── G020T004A024R021/
│   ├── g1/
│   │   ├── frame_00001.png
|   |   ├── frame_00002.png
|   |   ├── frame_00003.png
|   |   ├── frame_00004.png
|   |   ├── frame_00005.png
|   |   ├── frame_00006.png
│   │   ...
│   │      
│   └── nao/
|       ├── frame_00001.png
|          ...
```

First 

```
conda env create -f environment.yml
```


To train:

```
python -u tools/train.py configs/robots/$CONFIG --work-dir=./baselines/
```


To test:

```
python -u tools/test.py configs/robots/$CONFIG $CHECKPOINT --work-dir=./baselines/
```