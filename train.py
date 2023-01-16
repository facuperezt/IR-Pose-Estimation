import os
import sys


CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) 

sys.path.insert(0,os.path.join(BASE,'utils'))
sys.path.insert(0,os.path.join(BASE,'single_spot_table'))
from seg_makeh5 import processData, wirteFiles, write_data_label_hdf5

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
                     num_points = 2048):
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
        wirteFiles(path_aug, test_model_name='1170138_ROB_2')
        # wirte h5 format file
        write_data_label_hdf5(os.path.join(self.path_train,'train.txt'), self.path_dataset+'/seg_dataset_train_',2048)
        write_data_label_hdf5(os.path.join(self.path_train,'test.txt'), self.path_dataset+'/seg_dataset_test_',2048)
        # delete middle files
        os.system('rm -rf %s'%(path_aug))

    def train(self,
              log_dir,
              gpu = 0,
              num_point=2048,
              max_epoch = 100,
              batch_size = 16,
              learning_rate = 0.001):
        '''Train network
        
        Args:
            log_dir: path to the pn++ model
            gpu (int): GPU to use [default: 0]
            num_point (int): Point Number [default: 2048]
            max_epoch (int): Epoch to run [default: 100]
            batch_size (int): Batch Size during training [default: 16]
            learning_rate (float): Initial learning rate [default: 0.001]
        '''
        args = " --log_dir='"+log_dir+"' --gpu="+str(gpu)+" --num_point="+str(num_point)+" --max_epoch="+str(max_epoch)+\
            " --batch_size="+str(batch_size)+" --learning_rate="+str(learning_rate)
        path_to_train = './single_spot_table/seg_train.py'
        os.system('python '+path_to_train+args)
if __name__ == '__main__':
    train = TrainPointNet2(path_data='./data')
    train.train(log_dir='./data/seg_model')
    