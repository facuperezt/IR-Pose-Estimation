#%%
from utils.xml_parser import parse_frame_dump, list2array
import numpy as np
import sys
from utils.compatibility import listdir

def extract_frame_vectors(frame):
    tf = np.zeros((3,3))
    tf[0:3, 0] = frame[14:17].astype(float) # np.array([ 9.96926216e-01, -7.83461556e-02,  8.67287958e-17])
    tf[0:3, 1] = frame[17:20].astype(float) # np.array([ 0.05531376,  0.70384738, -0.70819436])
    tf[0:3, 2] = frame[20:23].astype(float) # np.array([0.05548431, 0.70601753, 0.70601753])

    return tf.T

def compare_frame_vectors(frame_one, frame_two):
    vec_one = extract_frame_vectors(frame_one)
    vec_two = extract_frame_vectors(frame_two)

    return np.linalg.norm(vec_one - vec_two)

def rotation_matrix_between_vectors(a,b):
    if (a == b).all() or (a == -b).all():
        return np.eye(3)
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
    vx = np.zeros((3,3))
    vx[0,1] = -v[2]
    vx[0,2] =  v[1]
    vx[1,2] = -v[0]
    vx[1,0] =  v[2]
    vx[2,0] = -v[1]
    vx[2,1] = v[0]

    return np.eye(3) + vx + np.dot(vx, vx) * ((1-c)/s**2)

def angles_from_rot_matrix(R):
    """
    Illustration of the rotation matrix / sometimes called 'orientation' matrix
    R = [ 
        R[0,0] , R[0,1] , R[0,2], 
        R[1,0] , R[1,1] , R[1,2],
        R[2,0] , R[2,1] , R[2,2]  
        ]

    REMARKS: 
    1. this implementation is meant to make the mathematics easy to be deciphered
    from the script, not so much on 'optimized' code. 
    You can then optimize it to your own style. 

    2. I have utilized naval rigid body terminology here whereby; 
    2.1 roll -> rotation about x-axis 
    2.2 pitch -> rotation about the y-axis 
    2.3 yaw -> rotation about the z-axis (this is pointing 'upwards') 
    """
    from math import (
        asin, pi, atan2, cos 
    )

    if R[2,0] != 1 and R[2,0] != -1: 
        pitch_1 = -1*asin(R[2,0])
        pitch_2 = pi - pitch_1 
        roll_1 = atan2( R[2,1] / cos(pitch_1) , R[2,2] /cos(pitch_1) ) 
        roll_2 = atan2( R[2,1] / cos(pitch_2) , R[2,2] /cos(pitch_2) ) 
        yaw_1 = atan2( R[1,0] / cos(pitch_1) , R[0,0] / cos(pitch_1) )
        yaw_2 = atan2( R[1,0] / cos(pitch_2) , R[0,0] / cos(pitch_2) ) 

        # IMPORTANT NOTE here, there is more than one solution but we choose the first for this case for simplicity !
        # You can insert your own domain logic here on how to handle both solutions appropriately (see the reference publication link for more info). 
        pitch = pitch_1 
        roll = roll_1
        yaw = yaw_1 
    else: 
        yaw = 0 # anything (we default this to zero)
        if R[2,0] == -1: 
            pitch = pi/2 
            roll = yaw + atan2(R[0,1],R[0,2]) 
        else: 
            pitch = -pi/2 
            roll = -1*yaw + atan2(-1*R[0,1],-1*R[0,2]) 

    # convert from radians to degrees
    roll = roll*180/pi 
    pitch = pitch*180/pi
    yaw = yaw*180/pi 

    rxyz_deg = [roll, pitch, yaw] 

    return rxyz_deg


def main(results_folder_path, model_name, verbose= False):
    FILE_ONE = f'{results_folder_path}/{model_name}/{model_name}.xml' # CHANGE HERE TO THE RIGHT FOLDER
    FILE_TWO = f'{results_folder_path}/{model_name}/{model_name}_predicted.xml'

    frames_one = list2array(parse_frame_dump(FILE_ONE, True))[:,3:]
    frames_two = list2array(parse_frame_dump(FILE_TWO, True))[:,3:]

    d = {}
    counter = {}
    for i in range(len(frames_one)):
        key = tuple(frames_one[i][10:13])
        for a,b in zip(frames_one[i][14:23].reshape(3,3).astype(float), frames_two[i][14:23].reshape(3,3).astype(float)):
            d[key] = d.get(key, np.zeros(3)) + np.array(angles_from_rot_matrix(rotation_matrix_between_vectors(b,a)))/len(frames_one)
            counter[key] = counter.get(key, 0) + 1
    if verbose:
        print(f"For Model: {model_name}:")
        for key,val in d.items():
            print(f"\tFor Rotation: {key}:")
            print(f"\t\tMean error angle: {[f'{axis}: {val:.2f}' for axis, val in zip(['x','y','z'], val)]} (#{counter[key]})\n")
    return d, counter
# %%
if __name__ == '__main__':
    models = listdir(sys.argv[1]) # bad practice for dayz 
    for model in models:
        main(sys.argv[1], model, verbose= True)
# %%
