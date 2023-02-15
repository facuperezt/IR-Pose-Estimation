# LookupTablePN
## Environment Configuration
An available docker can be found [here](https://hub.docker.com/repository/docker/chidianlizi/pointnet) with anaconda virtual environments named py37 and py27 already configured, where py37 is the python3 environment and py27 is the python2 environment. If you want to configure your environment locally, we also recommend using [Anaconda3](https://www.anaconda.com). After installing Anaconda, do the following steps one by one.


```bash
conda create -n py3 python=3.9.12
conda activate py3
pip install -r requirements.txt
conda deactivate py3
conda create -n py27 python=2.7.18
conda activate py27
conda install --channel https://conda.anaconda.org/marta-sd tensorflow-gpu=1.2.0
pip install -r requirements_py27.txt
conda deactivate py27
```



## Usage
Before starting, please place the files in the following directory format or use the dataset loader:
```
LookupTablePN
├── data
│   ├── train
│   │   ├── models
│   │   │   ├── componentname1
│   │   │   │   ├── componentname1.obj
│   │   │   │   ├── componentname1.mtl
│   │   │   │   ├── componentname1.xml
│   │   │   ├── componentname2
│   │   │   │   ├── componentname2.obj
│   │   │   │   ├── componentname2.mtl
│   │   │   │   ├── componentname2.xml
│   │   │   ├── ...
│   ├── torch
│   │   ├── MRW510_10GH.obj
│   │   ├── TAND_GERAD_DD.obj
│   ├── test
│   │   ├── models
│   │   │   ├── componentname1
│   │   │   │   ├── componentname1.obj
│   │   │   │   ├── componentname1.mtl
│   │   │   │   ├── componentname1.xml
│   │   │   ├── componentname2
│   │   │   │   ├── componentname2.obj
│   │   │   │   ├── componentname2.mtl
│   │   │   │   ├── componentname2.xml
│   │   │   ├── ...
```

### Training Step 0. Automatically load Dataset
(Requires the big folder 'Dataset' to be in the same directory)
(Skip if files are already in the right directory format)
In Python3 environment
```bash
python model_splitter.py <dataset> -t [*test_models]
```
#### Arguments
- dataset: Name of predetermined dataset **OR** path to folder. Assumes that all "\*.xml", "\*.obj" and "\*.mtl" files are in the same folder and organizes them into './data/' [^4]
- -t: The next entries are read as model names, that will **NOT** be used for training and are instead put into './data/test/models/' for inference.

[^4]: If './data/' already exists, it is renamed to './data_last/' to avoid unexpected behaviours.

### Training Step 1. Making lookup Table
In Python3 environment
```bash
python lut.py [-ARGUMENTS] [-FLAGS]
```
#### Arguments
- --label: 'HFD' for Geometrically labeled data, 'PDL' for pre-labeled data. (Default: 'PDL') [^1]
- --pcl_density: A parameter that controls the density of the point cloud, the smaller the value the higher the density (Default: 40) [^2]
- --crop_size: Edge lengths of point cloud slices in millimeters (Default: 400) [^2]
- --num_points: Number of points contained in the point cloud slice (Default: 2048) [^2]
- --free_cores: How many cores should remain unused.
#### Flags 
- --fast_sampling: meshes with high vertex density are uniformly sampled into pointclouds (faster than Poisson sampling).
- --decrease_lib: Removes redundant slices as post-processing (Very slow for big datasets)
- --skip_sampling: Skips sampling step.
- --skip_slicing: Skips slicing step.
- -s, --skip_both: Skips sampling and slicing.

[^1]: Labeling methods, there are 'PDL' and 'HFD'. PDL is pre-defined lable, if the color (material) of the parts of assemblies in the CAD design is differentiated according to the different kinds of parts, use PDL. HFD uses hybrid feature descriptors to cluster the parts of the assemblies, use it by running obj_geo_based_classification.py to generate the classification first, then enter the classification directory in arg hfd_path_classes

### Training Step 2. Pre-processing
In Python3 environment
```bash
python test.py -p [-ARGUMENTS]
```
#### Arguments
- --crop_size: Edge lengths of point cloud slices in millimeters (Default: 400) [^2]
- --num_points: Number of points contained in the point cloud slice (Default: 2048) [^2]


### Training Step 3. Train PN++
In Python2 environment
```bash
python train.py -t [-ARGUMENTS]
```
#### Arguments
- --gpu: Which GPU to use [0 or 1 for dual GPU setups] (Default: 0)
- --num_points: Number of points contained in the point cloud slice (Default: 2048) [^2]
- --max_epoch: Maximum amount of Epochs for training (Default: 100)
- --batch_size: Batch Size for training (Default: 16) [^2]
- --learning_rate: Initial Learning Rate (Default: 0.001)

### Testing Step 1. Data Pre-processing
In Python3 environment
```bash
python test.py -p [-ARGUMENTS]
```
#### Arguments
- --test_models: List of names of models to be used for inference. (If empty, all models in './data/test/models/' will be used.) [^3]
- --pcl_density: A parameter that controls the density of the point cloud, the smaller the value the higher the density (Default: 40) [^2]
- --crop_size: Edge lengths of point cloud slices in millimeters (Default: 400) [^2]
- --num_points: Number of points contained in the point cloud slice (Default: 2048) [^2]
- --batch_size: Batch Size for training (Default: 16) [^2]

#### Notes
The models need to be in './data/test/models/'

### Testing Step 2. Inference
In Python2 environment
```bash
python test.py -i [-ARGUMENTS]
```
#### Arguments
- --model_path: Path of model to run inference with (Default: ''./data/seg_model/model1.ckpt')
- --batch_size: Batch Size for training (Default: 16) [^2]

#### Outputs
Stores the results in './data/test/results/'


[^2]: The Arguments marked with [^2] must be coherent in all steps.
