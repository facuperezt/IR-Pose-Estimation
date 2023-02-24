#%%
import open3d as o3d
import numpy as np
import sys
from utils.foundation import load_pcd_data


if __name__ == '__main__':
    pcd = o3d.io.read_point_cloud(f'data/train/welding_zone/{sys.argv[1]}_{sys.argv[2]}.pcd')
    labels = load_pcd_data(f'data/train/welding_zone/{sys.argv[1]}_{sys.argv[2]}.pcd')[:, 3].astype(int)
    colors_choice = np.array([[1,0,0], [0,1,0], [0,0,1]])
    print(labels)
    colors = colors_choice[labels]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
# %%
