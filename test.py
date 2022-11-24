import os
import sys
import time

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(CURRENT_PATH) 

sys.path.insert(0,os.path.join(BASE,'utils'))
sys.path.insert(0,os.path.join(BASE,'single_spot_table'))
from test_preprocessing import sample_test_pc, slice_test

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
                  test_one_component,
                  batch_size):
        '''test a component
        
        Args:
            model_path: path to the pn++ model [default: './data/seg_model/model1.ckpt']
            test_input: path to the folder of welding slices for testing [default: './data/test/welding_zone_test']
            test_one_component: if only one component will be tested, enter the path here [default: None]
            batch_size: keep the same batch size with training
        '''
        args = " --model_path='"+model_path+"' --test_input="+test_input+" --test_one_component="+test_one_component\
            +" --batch_size="+str(batch_size)
        path_to_inference = './single_spot_table/seg_infer.py'
        os.system('python '+path_to_inference+args)
        
        
if __name__ == '__main__':
    te = PoseLookup(path_data='./data')
    # te.preprocessing('./data/test/models/22-10-14_Trailer')
    te.inference(model_path='./data/seg_model/model1.ckpt', test_input='./data/test/welding_zone_test', \
        test_one_component='./data/test/models/22-10-14_Trailer', batch_size=16)