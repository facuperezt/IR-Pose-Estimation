import sys
# Force python XML parser not faster C accelerators
# because we can't hook the C implementation
sys.modules['_elementtree'] = None
import xml.etree.ElementTree as ET
import numpy as np
from argparse import ArgumentParser
import os
from utils.xml_parser import list2array, parse_frame_dump
from utils.compatibility import listdir
from xml.dom.minidom import Document
import scipy.linalg as linalg
import time

def make_document(frames, parts_path, model_name):
    '''
    Combine the adapted array into one document
    '''
    torch_dict = {'0': 'MRW510_CDD_10GH', '1': 'TAND_GERAD_DD'}
    rot = frames[0][13:16].astype(float)*np.pi/180
    doc = Document()  # create DOM
    doc.version = "1.0"
    doc.encoding = "UTF-8"
    doc.standalone = "no"
    timestamp = 'Automatically created by TUB on: ' + time.strftime('%d.%m.%Y, %H:%M:%S')
    doc.appendChild(doc.createComment(timestamp))
    FRAME_DUMP = doc.createElement('FRAME-DUMP') # create root element
    FRAME_DUMP.setAttribute('VERSION', '1.0') 
    FRAME_DUMP.setAttribute('Baugruppe', model_name)
    doc.appendChild(FRAME_DUMP)
    prev_snaht_number = -1
    for frame in frames:
        if frame[-2] != prev_snaht_number:
            SNaht = doc.createElement('SNaht')
            SNaht.setAttribute('Name',str(frame[0]))
            SNaht.setAttribute('ID', str(frame[-1]) if frame[-1] is not None else None)
            SNaht.setAttribute('ZRotLock',str(frame[1]))
            SNaht.setAttribute('WkzWkl',str(frame[3]))
            SNaht.setAttribute('WkzName',str(frame[2]))
            FRAME_DUMP.appendChild(SNaht)

            Kontur = doc.createElement('Kontur')
            SNaht.appendChild(Kontur)

            Frames = doc.createElement('Frames')
            SNaht.appendChild(Frames)

            prev_snaht_number = frame[-2]

        Punkt = doc.createElement('Punkt')
        Punkt.setAttribute('X',str( frame[4]))
        Punkt.setAttribute('Y',str( frame[5]))
        Punkt.setAttribute('Z',str( frame[6]))
        Kontur.appendChild(Punkt)

        Fl_Norm1 = doc.createElement('Fl_Norm')
        Fl_Norm1.setAttribute('X',str( frame[7]))
        Fl_Norm1.setAttribute('Y',str( frame[8]))
        Fl_Norm1.setAttribute('Z',str( frame[9]))
        Punkt.appendChild(Fl_Norm1)

        Fl_Norm2 = doc.createElement('Fl_Norm')
        Fl_Norm2.setAttribute('X', str(frame[10]))
        Fl_Norm2.setAttribute('Y', str(frame[11]))
        Fl_Norm2.setAttribute('Z', str(frame[12]))
        Punkt.appendChild(Fl_Norm2)
        
        Rot = doc.createElement('Rot')
        Rot.setAttribute('X', str(frame[13]))
        Rot.setAttribute('Y', str(frame[14]))
        Rot.setAttribute('Z', str(frame[15]))
        Punkt.appendChild(Rot)
        EA = doc.createElement('Ext-Achswerte')
        EA.setAttribute('EA4', str(frame[16]))
        Punkt.appendChild(EA)
        if len(frame) > 20: # So we also include the flawed data, as in the original
            Frame = doc.createElement('Frame')
            Frames.appendChild(Frame)

            Pos = doc.createElement('Pos')
            Pos.setAttribute('X', str(frame[4]))
            Pos.setAttribute('Y', str(frame[5]))
            Pos.setAttribute('Z', str(frame[6]))
            Frame.appendChild(Pos)
            
            rot_matrix = linalg.expm(np.cross(np.eye(3), [1,0,0] / linalg.norm([1,0,0]) * (-rot[0])))

            xv = frame[17:20].astype(float)

            xv_r = np.matmul(rot_matrix, xv.T)
            XVek = doc.createElement('XVek')
            XVek.setAttribute('X', str(xv_r[0]))
            XVek.setAttribute('Y', str(xv_r[1]))
            XVek.setAttribute('Z', str(xv_r[2]))
            Frame.appendChild(XVek)
            try:
                yv = frame[20:23].astype(float)
            except ValueError as e:
                print(frame)
                print(frame[17:20])
                print(frame[20:23])
                print(frame[20])
                print(frame[21])
                print(frame[22])
                print(frame[23])
                print(frame[24])
                print(e)
                raise ValueError
            yv_r = np.matmul(rot_matrix, yv.T)
            YVek = doc.createElement('YVek')
            YVek.setAttribute('X', str(yv_r[0]))
            YVek.setAttribute('Y', str(yv_r[1]))
            YVek.setAttribute('Z', str(yv_r[2]))
            Frame.appendChild(YVek)
            zv = frame[23:26].astype(float)
            zv_r = np.matmul(rot_matrix, zv.T)
            ZVek = doc.createElement('ZVek')
            ZVek.setAttribute('X', str(zv_r[0]))
            ZVek.setAttribute('Y', str(zv_r[1]))
            ZVek.setAttribute('Z', str(zv_r[2]))
            Frame.appendChild(ZVek)
            Rot = doc.createElement('Rot')
            Rot.setAttribute('X', str(frame[13]))
            Rot.setAttribute('Y', str(frame[14]))
            Rot.setAttribute('Z', str(frame[15]))
            Frame.appendChild(Rot)
            EA = doc.createElement('Ext-Achswerte')
            EA.setAttribute('EA4', str(frame[16]))
            Frame.appendChild(EA)
    
    f = open(os.path.join(parts_path, model_name+'_predicted.xml'), 'wb')
    f.write(doc.toprettyxml(indent = '    ', encoding= "UTF-8")) #  removed standalone for compatibility with older python version
    f.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--original_xml_path', type=str, default='./data_last/test/models/Reisch', required= False, help='Path to the original models (i.e. ./data/test/models/)')
    parser.add_argument('--infered_points_folder_path', type=str, default='./data_last/test/results/', required= False, help='Path to results (i.e. ./data/test/results/)')

    args = parser.parse_args()

    assert os.path.isfile(os.path.join(args.original_xml_path)), 'Original .xml file not found.'
    assert os.path.isdir(os.path.join(args.infered_points_folder_path)), 'Infered files folder not found.'

    ret = {
        'file_path' : os.path.join(args.original_xml_path),
        'parts_path' : os.path.join(args.infered_points_folder_path),
    }

    return ret

def sort_xmls(parts_path, model_name):
    print(parts_path, model_name)
    all_parts = [a for a in listdir(parts_path) if len(os.path.splitext(a)[0].split('_')) > 1 and os.path.splitext(a)[-1] == '.xml' and not os.path.splitext(a)[0].split('_')[-1] == 'predicted' and model_name in a]
    tmp = sorted(all_parts, key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    print(tmp)
    return iter([os.path.join(parts_path, file) for file in tmp])


def main(original_xml_path, parts_path):
    """
    Reads the original XML, compares the predicted Frames for each point and merges them into the correct line on the base file.
    Then saves the resulting XML in the results folder.

    CAUTION: Only works for one element at a time.

    param: original_xml_path: path to the ground truth xml
    param: parts_path: path to the results folder, where the sub-XMLs are located
    """
    model_name = os.path.splitext(original_xml_path)[0].split('/')[-1]
    original_xml = list2array(parse_frame_dump(original_xml_path, safe_parsing= False))
    sorted_xmls = sort_xmls(parts_path, model_name)
    flag = True
    for i,pos in enumerate(original_xml): 
        if flag:
            xml_file = next(sorted_xmls)
            print(xml_file)
            next_xml = list2array(parse_frame_dump((xml_file)))[0] # some parts may get deleted during library decrease, but they will still be present in the original .xml
            flag = False
        if np.allclose(pos[4:7].astype(float), next_xml[4:7].astype(float)):
            original_xml[i][17:26] = next_xml[17:26].astype(np.float32)
            flag = True

    make_document(original_xml, parts_path, model_name)

if __name__ == '__main__':
    args = parse_args()
    original_xml_path = args['file_path']
    parts_path = args['parts_path']

    main(original_xml_path, parts_path)