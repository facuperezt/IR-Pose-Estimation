import os
import sys
import time

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) 

sys.path.insert(0,os.path.join(BASE,'utils'))
sys.path.insert(0,os.path.join(BASE,'single_spot_table'))
from test_preprocessing import sample_test_pc, slice_test
from utils.compatibility import listdir

from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

class PoseLookup():
    def __init__(self,
                 path_data):
        '''Lookup a best torch pose for test input
            
        Attributes:
            path_data: The path to the data folder, see readme for the specific directory structure
        '''
        self.path_data = path_data
        self.path_test = os.path.join(self.path_data, 'test')
        self.path_models = os.path.join(self.path_test, 'models')
        self.path_dataset = os.path.join(self.path_test, 'dataset')
        if not os.path.exists(self.path_dataset):
            os.makedirs(self.path_dataset)

    def preprocessing_pool(self, args):
        path_test_components, pcl_density, crop_size, num_points = args
        for path in path_test_components:
            self.preprocessing(path, pcl_density, crop_size, num_points)

    def preprocessing(self,
                      path_test_component,
                      pcl_density = 40,
                      crop_size = 400,
                      num_points = 2048):
        '''Sampling and slicing for test component
        
        Args:
            path_test_component: path to the test component
            pcl_density: A parameter that controls the density of the point cloud, the smaller 
                         the value the higher the density
            crop_size: Edge lengths of point cloud slices in millimeters. Must be consistent
                            with the values used in the previous lookup table creation [default: 400]
            num_points: Number of points contained in the point cloud slice  [default: 2048]
        '''
        component = os.path.split(path_test_component)[-1]
        files = os.listdir(path_test_component)
        for file in files:
            if os.path.splitext(file)[-1] == '.obj':
                old_name = os.path.splitext(file)[0]
                if not old_name == component:
                    os.rename(os.path.join(self.path_models,component,file),os.path.join(self.path_models,component,component+'.obj'))
                file_data = ''
                with open(os.path.join(self.path_models,component,component+'.obj'), 'r') as f:                        
                    for line in f:
                        if '.mtl' in line:
                            file_data += 'mtllib '+component+'.mtl\n'
                        else:                                
                            file_data += line                    
                with open(os.path.join(self.path_models,component,component+'.obj'), 'w') as f:
                    f.write(file_data)
            if os.path.splitext(file)[-1] == '.xml':
                old_name = os.path.splitext(file)[0]
                if not old_name == component:
                    os.rename(os.path.join(self.path_models,component,file),os.path.join(self.path_models,component,component+'.xml'))
            if os.path.splitext(file)[-1] == '.mtl':
                old_name = os.path.splitext(file)[0]
                if not old_name == component:
                    os.rename(os.path.join(self.path_models,component,file),os.path.join(self.path_models,component,component+'.mtl'))

        print ('sampling... ...', component)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        sample_test_pc(path_test_component, pcl_density)
        path_pc = os.path.join(self.path_models, component, component+'.xyz')   
        path_xml = os.path.join(self.path_models, component, component+'.xml')
        path_wztest = os.path.join(self.path_test, 'welding_zone_test', component)
        if not os.path.exists(path_wztest):
            os.makedirs(path_wztest)
            print ('slicing... ...', component)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            slice_test(path_pc, path_xml, path_wztest, crop_size, num_points)
    
    def inference(self,
                  model_path,
                  test_input,
                  test_one_component = None,
                  batch_size = 16):
        '''test a component
        
        Args:
            model_path: path to the pn++ model [default: './data/seg_model/model1.ckpt']
            test_input: path to the folder of welding slices for testing [default: './data/test/welding_zone_test']
            test_one_component: if only one component will be tested, enter the path here [default: None]
            batch_size: keep the same batch size with training
        '''
        if test_one_component is not None:
            args = " --model_path='"+model_path+"' --test_input="+test_input+" --test_one_component="+test_one_component\
                +" --batch_size="+str(batch_size)
        else:
            args = " --model_path='"+model_path+"' --test_input="+test_input+" --batch_size="+str(batch_size)
        path_to_inference = './single_spot_table/seg_infer.py'
        os.system('python '+path_to_inference+args)
        
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-p', '--preprocess', action='store_true', help='Run script preprocessing (Requires python 3.x)')
    parser.add_argument('-i', '--inference', action='store_true', help='Run inference script (Requires python 2.x)')
    parser.add_argument('--test_models', nargs='*', type=str, help='Pass name of models to be preprocessed for inference. If no name is passed all models in --test_folder will be preprocessed.')
    parser.add_argument('--pcl_density', type=int, default= 40, help='Pointcloud Density (Must be the same as in training)')
    parser.add_argument('--crop_size', type=int, default= 400, help='Cropped Slice Size (Must be the same as in training)')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per PCL (Must be the same as in training)')
    parser.add_argument('--batch_size', type= int, default= 16, help='Batch size of network (Must be the same as in training)')
    parser.add_argument('--model_path', type=str, default= './data/seg_model/model1.ckpt', help='Path of model to run inference with')
    args = parser.parse_args()
    assert args.preprocess ^ args.inference, 'Script has to be used with -p OR -i flag'

    te = PoseLookup(path_data='./data')
    if args.preprocess:
        assert sys.version[0] == '3', 'Python 3.x is required for preprocessing (-p)'
        if args.test_models is None or len(args.test_models) != 1:
            path_test = './data/test/models/'
            test_models = listdir(path_test)
            if args.test_models is not None and len(args.test_models) > 0:
                test_models = [tm for tm in test_models if tm in args.test_models]
            test_models = [path_test + test_model for test_model in test_models if os.path.isdir(os.path.join(path_test, test_model))]    # remove non-folders
            nr_processes = max(min(len(test_models), cpu_count() - 2), 1)
            k, m = divmod(len(test_models), nr_processes)                                                    # divide among processors
            split_paths = list(test_models[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(nr_processes))
            pcl_density, crop_size, num_points = args.pcl_density, args.crop_size, args.num_points
            repeated_args = [[pcl_density, crop_size, num_points]]*nr_processes
            all_args = [args.insert(0, path) for path, args in zip(split_paths, repeated_args)]
            print ('preprocessing test models... ', nr_processes, ' workers ...', test_models)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            with Pool(nr_processes) as p:
                p.map(te.preprocessing_pool, [_args for _args in all_args])

            print('processing finished')
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            # for test_model in test_models:
            #     te.preprocessing(path_test_component='./data/test/models/' + test_model, pcl_density=40, crop_size=400, num_points=2048)
        elif len(args.test_models) == 1 in listdir('./data/test/models'):
            te.preprocessing(path_test_component='./data/test/models/' + args.test_models[0], pcl_density=args.pcl_density, crop_size=args.crop_size, num_points=args.num_points)
    elif args.inference:
        assert sys.version[0] == '2', 'Python 2.x is required for preprocessing (-i)'
        te.inference(model_path=args.model_path, test_input='./data/test/welding_zone_test', batch_size=args.batch_size)