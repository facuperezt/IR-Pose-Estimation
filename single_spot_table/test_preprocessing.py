import os
import open3d as o3d
import numpy as np
from xml.dom.minidom import Document
from copy import copy
import pickle
import math
import sys
import copy
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT,'utils'))
from xml_parser import parse_frame_dump, list2array
from foundation import points2pcd, load_pcd_data, fps
from math_util import rotate_mat, rotation_matrix_from_vectors

def sample_test_pc(path, density=40):
    '''Convert mesh to pointcloud without labels
    
    '''
    namestr = os.path.split(path)[-1]
    files = os.listdir(path)
    for file in files:
        if file == namestr+'.obj':
            mesh = o3d.io.read_triangle_mesh(os.path.join(path, file))                 
            number_points = int(mesh.get_surface_area()/density) # get number of points according to surface area
            pc = mesh.sample_points_poisson_disk(number_points, init_factor=5) # poisson disk sampling
            xyz = np.asarray(pc.points)
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(xyz)
            o3d.io.write_point_cloud(os.path.join(path, namestr+'.xyz'),pc)


class WeldScene_test:
    '''
    create welding scene for slicing
    '''
    def __init__(self, pc_path):
        self.pc = o3d.io.read_point_cloud(pc_path)
        self.xyz = np.asarray(self.pc.points)
        self.CROP_EXTENT = 100
        self.CROP_LEN = 400

    def crop(self, weld_info, crop_size=400, num_points=2048, vis=False):
        pc = copy.copy(self.pc)

        norm1 = np.around(weld_info[4:7], decimals=6)
        norm2 = np.around(weld_info[7:10], decimals=6)
        # print (norm1, norm2)
        
        rot = weld_info[10:13]*math.pi/180
        rotation = rotate_mat(axis=[1,0,0], radian=rot[0])
        tf = np.zeros((4,4))           
        tf[3,3] = 1.0
        tf[0:3,0:3] = rotation
        norm1_r = np.matmul(rotation, norm1.T)
        norm2_r = np.matmul(rotation, norm2.T)
        extent = crop_size-10
        crop_extent = np.array([extent, extent, extent])
        weld_spot = weld_info[1:4]
        pc.translate(-weld_spot)

        norm_ori = np.array([0, 0, 1])
        R = rotation_matrix_from_vectors(norm_ori, norm_ori)
        bbox = o3d.geometry.OrientedBoundingBox(center=(extent/2-60)*norm1_r+(extent/2-60)*norm2_r , R=R, extent=crop_extent)
        coor_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=np.array([0,0,0]))

        pc.transform(tf)
        xyz = np.asarray(pc.points)
        cropped_pc = pc.crop(bbox)
        idx_crop = bbox.get_point_indices_within_bounding_box(pc.points)
        xyz_crop = xyz[idx_crop]
        xyz_crop = np.unique(xyz_crop, axis=0)
        while xyz_crop.shape[0] < num_points:
            xyz_crop = np.vstack((xyz_crop, xyz_crop))
        xyz_crop = fps(xyz_crop, num_points)
        if vis:
            o3d.visualization.draw_geometries([cropped_pc, coor_world, bbox])


        return xyz_crop, cropped_pc

def slice_test(pc_path, path_xml, path_dist, crop_size, num_points):
    '''Create test slices with welding info without ground truth pose
    
    '''
    Scene = WeldScene_test(pc_path)
    namestr = os.path.splitext(os.path.split(path_xml)[-1])[0]
    print (namestr)
    # read XML file
    frames = list2array(parse_frame_dump(path_xml))
    for i in range(frames.shape[0]):
        weld_info = frames[i,3:].astype(float)
        cxyz, _ = Scene.crop(weld_info, crop_size=crop_size, num_points=num_points)
        cpc = o3d.geometry.PointCloud()
        cpc.points = o3d.utility.Vector3dVector(cxyz)
        # coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=np.array([0,0,0]))
        # o3d.visualization.draw_geometries([cpc, coor])    
        o3d.io.write_point_cloud(path_dist + '/' + namestr + '_' + str(i) +'.xyz', cpc, True)
        doc = Document()  # create DOM
        FRAME_DUMP = doc.createElement('FRAME-DUMP') # create root element
        FRAME_DUMP.setAttribute('VERSION', '1.0')
        FRAME_DUMP.setAttribute('Baugruppe', namestr + '_' + str(i))
        doc.appendChild(FRAME_DUMP)
        # if weld_info[0] == 0:
        #     torch = 'MRW510_CDD_10GH'
        # elif weld_info[0] == 1:
        #     torch = 'TAND_GERAD_DD'
        # else:
        #     torch = ''
            
        SNaht = doc.createElement('SNaht')
        SNaht.setAttribute('Name',frames[i,0])
        SNaht.setAttribute('ZRotLock',frames[i,1])
        SNaht.setAttribute('WkzName',frames[i,2])
        SNaht.setAttribute('WkzWkl',frames[i,3])

        FRAME_DUMP.appendChild(SNaht)

        Kontur = doc.createElement('Kontur')
        SNaht.appendChild(Kontur)

        Punkt = doc.createElement('Punkt')
        Punkt.setAttribute('X', str(frames[i,4]))
        Punkt.setAttribute('Y', str(frames[i,5]))
        Punkt.setAttribute('Z', str(frames[i,6]))
        Kontur.appendChild(Punkt)

        Fl_Norm1 = doc.createElement('Fl_Norm')
        Fl_Norm1.setAttribute('X', str(frames[i,7]))
        Fl_Norm1.setAttribute('Y', str(frames[i,8]))
        Fl_Norm1.setAttribute('Z', str(frames[i,9]))
        Punkt.appendChild(Fl_Norm1)

        Fl_Norm2 = doc.createElement('Fl_Norm')
        Fl_Norm2.setAttribute('X', str(frames[i,10]))
        Fl_Norm2.setAttribute('Y', str(frames[i,11]))
        Fl_Norm2.setAttribute('Z', str(frames[i,12]))
        Punkt.appendChild(Fl_Norm2)

        Rot = doc.createElement('Rot')
        Rot.setAttribute('X', str(frames[i,13]))
        Rot.setAttribute('Y', str(frames[i,14]))
        Rot.setAttribute('Z', str(frames[i,15]))
        Punkt.appendChild(Rot)
        
        EA3 = doc.createElement('Ext-Achswerte')
        EA3.setAttribute('EA3', str(frames[i,16]))
        Punkt.appendChild(EA3)
  
        f = open(path_dist + '/' + namestr+'_'+str(i)+'.xml','w')
        f.write(doc.toprettyxml(indent = '  '))
        f.close()

if __name__ == '__main__':
    # rewrite_xml()
    ################sort out#######################
    # test_path = '../data/test_ds2/models'
    # src = '../data/welding_objects_ds2'
    # files = os.listdir(src)
    
    # objs = []
    # for file in files:
    #     name = os.path.splitext(file)[0]
    #     if name not in objs:
    #         objs.append(name)
    # print (objs)
    # for name in objs:
    #     os.mkdir(os.path.join(test_path,name))
    #     for file in files:
    #         if os.path.splitext(file)[0] == name:
    #             os.system ("cp %s %s" % (os.path.join(src, file), os.path.join(test_path, name)))
                
    
    ################rotataion&slice################
    path_test = '../data/test/models'
    test_files = os.listdir(path_test)
    for test_file in test_files:
        print ('sampling... ...', test_file)
        print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        sample_test_pc(os.path.join(path_test, test_file))
        path_pc = os.path.join(path_test, test_file, test_file+'.xyz')   
        name = test_file
        path_xml = os.path.join(path_test, name, name+'.xml')
        path_dist = os.path.join('../data/test/welding_zone_test', name)
        if not os.path.exists(path_dist):
            os.makedirs(path_dist)
            print ('slicing... ...', test_file)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
            slice_test(path_pc, path_xml, path_dist)
    