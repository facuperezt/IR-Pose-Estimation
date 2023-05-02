# LookupTablePN

## Requirements
* [Anaconda3](https://www.anaconda.com)
* CUDA-enabled GPU


## Setup
For installing the Environment we provide a simple bash file 

```bash
git clone --recurse-submodules git@github.com:ignc-research/IR-Pose-Estimation.git
cd IR-Pose-Estimation
source install_openpoints.sh
```

CUDA-11.7 will be automatically installed in the conda environment. If you want to use a different cuda version modify `install_openpoint.sh`.

## Two liner usage
On Python3 environment
```python
python prepare.py
```
On Python2 environment
```python
python run.py
```

"prepare.py" will read a Dataset from Datasets and pre-process it.
"run.py" will train a model on the small Dataset for 5 epochs and run inference on it.

If this runs you're probably all set.


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
### Model Selection

For Training and Testing you can choose between different models with are saved in configuration files './cfgs/'
Currently there is a model for PointNet++ and PointNext to choose from. However, with OpenPoints it is easy to create new models by creating new config file. For that please read the Documentation [here](https://guochengqian.github.io/PointNeXt/)

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

#### In the case of unlabeled Data
In Python3 environment
```bash
python single_spot_table/obj_geo_based_classification.py 
```

And follow the instructions given by the program to label the dataset using its geometrical properties

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
python train.py -p [-ARGUMENTS]
```
#### Arguments
- --crop_size: Edge lengths of point cloud slices in millimeters (Default: 400) [^2]
- --num_points: Number of points contained in the point cloud slice (Default: 2048) [^2]


### Training Step 3. Train via OpenPoints
For example, for training with PointNext run 
```bash
CUDA_VISIBLE_DEVICES=0 python ./train_openpoints.py --cfg cfgs/PoseEstimation/pointnext-s.yaml --mode train
```

For training a different model just change the Argument for `--cfg`.

*WARNING: Depending on the number of classes you have in you data, you need change the parameter NUM_CLASSES in the cfg-files.*

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
For example, for inference with PointNext run 

```bash
CUDA_VISIBLE_DEVICES=0 python ./train_openpoints.py --cfg cfgs/PoseEstimation/pointnext-s.yaml --mode inference --pretrained_path /path/to/your/pretrained_model
```

For test a different model just change the Argument for `--cfg`.

#### Outputs
Stores the results in './data/test/results/'

## Note
Some of the features from OpenPoints can not be used in the current version of PoseEstimation, e.g. running on multiple GPUs.



[^2]: The Arguments marked with [^2] must be coherent in all steps.
