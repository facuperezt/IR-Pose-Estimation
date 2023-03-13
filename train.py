import os
import sys


CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) 

sys.path.insert(0,os.path.join(BASE,'utils'))
sys.path.insert(0,os.path.join(BASE,'single_spot_table'))
from seg_makeh5 import processData, wirteFiles, write_data_label_hdf5

from argparse import ArgumentParser

class TrainPointNet2():
    '''Train the pn++ semantic segmentation network 
    making h5 format dataset, training
        
    Attributes:
        path_data: The path to the data folder, see readme for the specific directory structure
    '''
    def __init__(self,
                 path_data):
        self.path_data = path_data
        self.path_train = os.path.join(self.path_data, 'train')
        self.path_dataset = os.path.join(self.path_train, 'dataset')
        if not os.path.exists(self.path_dataset):
            os.makedirs(self.path_dataset)
    
    def make_dataset(self,
                     crop_size = 400,
                     num_points = 2048,
                     test_model_name = None):
        '''Make h5 format dataset
        
        Args:
            crop_size (int): Edge lengths of point cloud slices in millimeters. Must be consistent
                            with the values used in the previous lookup table creation [default: 400]
            num_points (int): Number of points contained in the point cloud slice  [default: 2048]
        '''
        path_wzc = os.path.join(self.path_train, 'welding_zone_comp')  
        path_aug = os.path.join(self.path_train,'aug')
        if not os.path.exists(path_aug):
            os.makedirs(path_aug)
        
        # random scale and augmentation     
        processData(path_wzc, path_aug, crop_size, num_points)
        # split trainset and testset
        wirteFiles(path_aug, test_model_name=test_model_name)
        # wirte h5 format file
        write_data_label_hdf5(os.path.join(self.path_train,'train.txt'), self.path_dataset+'/seg_dataset_train_',2048)
        write_data_label_hdf5(os.path.join(self.path_train,'test.txt'), self.path_dataset+'/seg_dataset_test_',2048)
        # delete middle files
        os.system('rm -rf %s'%(path_aug))

    def train(self):
        '''Train network
        
        Args:
            log_dir: path to the pn++ model
            gpu (int): GPU to use [default: 0]
            num_point (int): Point Number [default: 2048]
            max_epoch (int): Epoch to run [default: 100]
            batch_size (int): Batch Size during training [default: 16]
            learning_rate (float): Initial learning rate [default: 0.001]
        '''
        args = " --cfg cfgs/PoseEstimation/pointnext-s.yaml"
        path_to_train = './train_openpoints.py'
        os.system('CUDA_VISIBLE_DEVICES=1 python '+path_to_train+args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--preprocess', action='store_true', help='Prepare the dataset for training (Python 3.x required)')
    parser.add_argument('-t', '--train', action='store_true', help='Train with prepared dataset (Python 2.x required)')
    parser.add_argument('--pcl_density', type=int, default= 40, help='Pointcloud Density (Must be the same for testing)')
    parser.add_argument('--crop_size', type=int, default= 400, help='Cropped Slice Size (Must be the same for testing)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per PCL (Must be the same for testing)')
    parser.add_argument('--max_epoch', type=int, default=100, help='Maximum amount of Epochs for training')
    parser.add_argument('--batch_size', type=int, default= 16, help='Batch Size for training')
    parser.add_argument('--learning_rate', type=float, default= 0.001, help='Learning Rate for training')
    parser.add_argument('--gpu', type= int, default=0, choices=[0,1], help='Which GPU to use.')
    args = parser.parse_args()
    assert args.preprocess ^ args.train, 'Script must be called with -p OR -t flag.'

    tr = TrainPointNet2(path_data='./data')
    if args.preprocess:
        assert sys.version[0] == '3', 'Preprocessing requires Python 3.x (-p)'
        # make dataset
        tr.make_dataset(crop_size=args.crop_size, num_points=args.num_points)
    elif args.train:
        # training
        tr.train()

    