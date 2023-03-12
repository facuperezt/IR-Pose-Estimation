import argparse
import glob
import logging
import numpy as np
import os
import yaml
import pickle
import os
import numpy as np
import argparse
from sklearn.cluster import DBSCAN
import open3d as o3d
import pickle
from xml.dom.minidom import Document
import copy
import time
import random
import sys
import scipy.linalg as linalg
import math
#from hdf5_util import *
from utils.foundation import draw, fps, load_pcd_data, fps
from utils.xml_parser import list2array, parse_frame_dump
from utils.math_util import rotate_mat

import torch
from tqdm import tqdm

import h5py

from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils

from single_spot_table import seg_infer

from single_spot_table.compatibility import listdir

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)

batch_size = 16
INPUT_PATH = './data/test/welding_zone_test'
TEST_COMP = None

f = open('./data/train/parts_classification/label_dict.pkl', 'rb')
lable_list = pickle.load(f)
labeldict = dict([val, key] for key, val in lable_list.items()) 

def main(gpu, cfg):
    """if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
        """

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    """if cfg.rank == 0:
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    else:
        writer = None"""
    writer = None

    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    """if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info('Using Distributed Data parallel ...')
    """

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate

    # optionally resume from a checkpoint
    #TODO: Inference
    model_module = model.module if hasattr(model, 'module') else model
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
        else:
            if cfg.mode == 'test':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                data_list = generate_data_list(cfg)
                logging.info(f"length of test dataset: {len(data_list)}")
                logging.info(f"class types: {type(model)}")
                logging.info(f"class types: {type(data_list)}")
                logging.info(f"class types: {type(cfg)}")
                test_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='test',
                                             distributed=cfg.distributed,
                                             )
                test_miou, test_macc, test_oa, test_ious, test_accs, _ = test(model, data_list, cfg)

                if test_miou is not None:
                    with np.printoptions(precision=2, suppress=True):
                        logging.info(
                            f'Best ckpt @E{best_epoch},  test_oa , test_macc, test_miou: {test_oa:.2f} {test_macc:.2f} {test_miou:.2f}, '
                            f'\niou per cls is: {test_ious}')
                    cfg.csv_path = os.path.join(cfg.run_dir, cfg.run_name + '_test.csv')
                    #write_to_csv(test_oa, test_macc, test_miou, test_ious, best_epoch, cfg)
                return test_miou

            elif cfg.mode == 'inference':
                best_epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                seg_inference(model)
                return None
                #inf_loader = build_dataloader_from_cfg(cfg.batch_size,
                #                            cfg.dataset,
                #                            cfg.dataloader,
                #                            datatransforms_cfg=cfg.datatransforms,
                #                            split='test',
                #                            distributed=cfg.distributed,
                #                           )
                #test_miou, test_macc, test_oa, test_ious, test_accs, _ = inference(model, inf_loader, cfg)
                #return test_miou 
                
        """elif 'encoder' in cfg.mode:
            logging.info(f'Finetuning from {cfg.pretrained_path}')
            load_checkpoint(model_module.encoder, cfg.pretrained_path, cfg.get('pretrained_module', None))
        else:
            logging.info(f'Finetuning from {cfg.pretrained_path}')
            load_checkpoint(model, cfg.pretrained_path, cfg.get('pretrained_module', None))
        """

    else:
        logging.info('Training from scratch')

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    cfg.criterion_args.weight = None
    if cfg.get('cls_weighed_loss', False):
        if hasattr(train_loader.dataset, 'num_per_class'):
            cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
        else:
            logging.info('`num_per_class` attribute is not founded in dataset')
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    val_miou, val_macc, val_oa, val_ious, val_accs = 0., 0., 0., [], []
    best_val, macc_when_best, oa_when_best, ious_when_best, best_epoch = 0., 0., 0., [], 0
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        train_loss, train_miou, train_macc, train_oa, _, _ = \
            train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg)

        is_best = False
        if epoch % cfg.val_freq == 0:
            val_miou, val_macc, val_oa, val_ious, val_accs = validate_fn(model, val_loader, cfg)
            if val_miou > best_val:
                is_best = True
                best_val = val_miou
                macc_when_best = val_macc
                oa_when_best = val_oa
                ious_when_best = val_ious
                best_epoch = epoch
                with np.printoptions(precision=2, suppress=True):
                    logging.info(
                        f'Find a better ckpt @E{epoch}, val_miou {val_miou:.2f} val_macc {macc_when_best:.2f}, val_oa {oa_when_best:.2f}'
                        f'\nmious: {val_ious}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_miou {train_miou:.2f}, val_miou {val_miou:.2f}, best val miou {best_val:.2f}')
#        if writer is not None:
#            writer.add_scalar('best_val', best_val, epoch)
#            writer.add_scalar('val_miou', val_miou, epoch)
#            writer.add_scalar('macc_when_best', macc_when_best, epoch)
#            writer.add_scalar('oa_when_best', oa_when_best, epoch)
#            writer.add_scalar('val_macc', val_macc, epoch)
#            writer.add_scalar('val_oa', val_oa, epoch)
#            writer.add_scalar('train_loss', train_loss, epoch)
#            writer.add_scalar('train_miou', train_miou, epoch)
#            writer.add_scalar('train_macc', train_macc, epoch)
#            writer.add_scalar('lr', lr, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best
                            )
            is_best = False

    # validate
    with np.printoptions(precision=2, suppress=True):
        logging.info(
            f'Best ckpt @E{best_epoch},  val_oa {oa_when_best:.2f}, val_macc {macc_when_best:.2f}, val_miou {best_val:.2f}, '
            f'\niou per cls is: {ious_when_best}')

    #if writer is not None:
    #    writer.close()

   #wandb.finish(exit_code=True)


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, data in pbar:

        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        num_iter += 1
        target = data['y'].squeeze(-1)
        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            logits = model(data)
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])

        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), target)
        loss_meter.update(loss.item())

        if idx % cfg.print_freq:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    miou, macc, oa, ious, accs = cm.all_metrics()
    return loss_meter.avg, miou, macc, oa, ious, accs

@torch.no_grad()
def validate(model, val_loader, cfg, num_votes=1, data_transform=None):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)

        target = data['y'].squeeze(-1)
        
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        logits = model(data)
        #print(logits)
        #print(type(logits))
        #print(logits.shape)
        
        if 'mask' not in cfg.criterion_args.NAME or cfg.get('use_maks', False):
            #print("Arg: ",logits.argmax(dim=1))
            #print(logits.argmax(dim=1).shape)
            cm.update(logits.argmax(dim=1), target)
        else:
            mask = data['mask'].bool()
            cm.update(logits.argmax(dim=1)[mask], target[mask])
        #raise Exception    

        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        coord = data['pos'].cpu().numpy()[0]
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        label = target[0].cpu().numpy()
        if cfg.ignore_index is not None:
            if (label == cfg.ignore_index).sum() > 0:
                pred[label == cfg.ignore_index] = cfg.num_classes
                label[label == cfg.ignore_index] = cfg.num_classes
        vis_multi_points([coord, coord], labels=[label, pred])
        """
        # tp, union, count = cm.tp, cm.union, cm.count
        # if cfg.distributed:
        #     dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        # miou, macc, oa, ious, accs = get_mious(tp, union, count)
        # with np.printoptions(precision=2, suppress=True):
        #     logging.info(f'{idx}-th cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
        #                 f'\niou per cls is: {ious}')

    tp, union, count = cm.tp, cm.union, cm.count
    miou, macc, oa, ious, accs = get_mious(tp, union, count)
    return miou, macc, oa, ious, accs

#def generate_data_list(cfg):
#    if 'poseestimation' in cfg.dataset.common.NAME.lower():
#        my_file = open("./data/train/test.txt", "r")
#        data = my_file.read()
#        data_list = data.split("\n")
#        my_file.close()
#        print("Data list:" ,data_list)
#    else:
#        raise Exception('dataset not supported yet'.format(args.data_name))
#    return data_list

def load_data(data_path, cfg):
    label, feat = None, None
    if 's3dis' in cfg.dataset.common.NAME.lower():
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
        feat = np.clip(feat / 255., 0, 1).astype(np.float32)
    elif 'scannet' in cfg.dataset.common.NAME.lower():
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat = data[0], data[1]
        if cfg.dataset.test.split != 'test':
           label = data[2]
        else:
            label = None
        feat = np.clip((feat + 1) / 2., 0, 1).astype(np.float32)
    elif 'semantickitti' in cfg.dataset.common.NAME.lower():
        coord = load_pc_kitti(data_path[0])
        if cfg.dataset.test.split != 'test':
            label = load_label_kitti(data_path[1], remap_lut_read)
    coord -= coord.min(0)

    idx_points = []
    voxel_idx, reverse_idx_part,reverse_idx_sort = None, None, None
    voxel_size = cfg.dataset.common.get('voxel_size', None)

    if voxel_size is not None:
        # idx_sort: original point indicies sorted by voxel NO.
        # voxel_idx: Voxel NO. for the sorted points
        idx_sort, voxel_idx, count = voxelize(coord, voxel_size, mode=1)
        if cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor':
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_part = idx_sort[idx_select]
            npoints_subcloud = voxel_idx.max()+1
            idx_shuffle = np.random.permutation(npoints_subcloud)
            idx_part = idx_part[idx_shuffle] # idx_part: randomly sampled points of a voxel
            reverse_idx_part = np.argsort(idx_shuffle, axis=0) # revevers idx_part to sorted
            idx_points.append(idx_part)
            reverse_idx_sort = np.argsort(idx_sort, axis=0)
        else:
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                np.random.shuffle(idx_part)
                idx_points.append(idx_part)
    else:
        idx_points.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx_sort

@torch.no_grad()
def inference(model, inf_loader, cfg):
    """_summary_

    Args:
        model (_type_): _description_
        inf_loader (_type_): _description_
        cfg (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.eval()
    set_random_seed(0)
    cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    pbar = tqdm(enumerate(inf_loader), total=inf_loader.__len__(), desc='Inf')
    for idx, data in pbar:
        #print(data['pos'].shape)
        keys = data.keys() if callable(data.keys) else data.keys
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
        #target = data['y'].squeeze(-1)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        logits = model(data)
        #print(logits)
    return None, None, None, None, None, None

@torch.no_grad()     
def seg_inference(model):
    model.to(torch.device("cuda"))
    infer_all_sep(model, TEST_COMP)
    path = './data/test/results'
    folders = listdir(path)
    for folder in folders: # This loop might not be needed anymore.
        files = listdir(os.path.join(path, folder))
        xml_list = []
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                xml_list.append(os.path.join(path, folder, file))
        with open(os.path.join(path, folder, folder +'.xml'), 'w+') as f:
            f.write('<?xml version="1.0" encoding="utf-8" standalone="no" ?>\n')
            f.write('<frame-dump version="1.0" baugruppe="'+folder+'">\n')
            for xml in xml_list:
                g = open(xml, 'r')
                context = g.readlines()
                for line in context[2:-1]:
                    f.write(line)
            f.write('</frame-dump>')
    
    if not TEST_COMP == None:
        folders = [os.path.split(TEST_COMP)[-1]]
    else:
        folders = listdir(INPUT_PATH)
    for folder in folders:
        print(os.path.join(INPUT_PATH, '../models', folder, folder+'.xml'))
        print(path)
        print(os.getcwd())
        #os.system('python ' + 'update_xml.py ' + '--original_xml_path='+os.path.join(INPUT_PATH, '../models', folder, folder+'.xml') + ' --infered_points_folder_path='+path+'/'+folder)

@torch.no_grad()
def infer_all_sep(model_1, path_test_component=None):
    with open('./data/train/lookup_table/lookup_table.pkl', 'rb') as f:
        dict_all = pickle.load(f)
    with open('./data/ss_lookup_table/norm_fd.pkl', 'rb') as g:
        norm_fd = pickle.load(g)
    model = model_1
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6)) #TODO: Replace model
    if not path_test_component == None:
        folders = [os.path.split(path_test_component)[-1]]
    else:
        folders = listdir(INPUT_PATH)
    for folder in folders:
        if not os.path.exists('./data/test/results/'+folder):
            os.makedirs('./data/test/results/'+folder)
        else:
            continue
        match_dict = {}
        # get the test slices of a component
        slices_path = os.path.join(INPUT_PATH, folder)
        files = listdir(slices_path)
        #print(folder)
        #print(len(files))
        t = []
        num_t = 0
        for file in files:
            if os.path.splitext(file)[1] == '.xyz':
                start = time.time()
                namestr = os.path.splitext(file)[0]
                print('Current input file: ', file)
                file_path = os.path.join(slices_path, file)
                pc = o3d.io.read_point_cloud(file_path)
                
                coor_world = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200, origin=np.array([0,0,0]))
                src_xml = os.path.join(INPUT_PATH, folder, namestr+'.xml')
                
                frames = list2array(parse_frame_dump(src_xml))
                tor = frames[0][3].astype(float)
                normals_1 = frames[0][7:10].astype(float)
                normals_2 = frames[0][10:13].astype(float)

                rot = frames[0][13:16].astype(float)*math.pi/180
                rotation = rotate_mat(axis=[1,0,0], radian=rot[0])
                norm1_r = np.matmul(rotation, normals_1.T)
                norm2_r = np.matmul(rotation, normals_2.T)
                # normals_r = np.hstack((norm1_r, norm2_r))
                normals_r = norm1_r + norm2_r
                norm_r = np.round(normals_r.astype(float), 2)
                norm_s = ''
                for i in norm_r:
                    if i == 0:
                        norm_s += str(0.0)
                    else:
                        norm_s += str(i)
                xyz = np.asarray(pc.points)
                #print(type(xyz))
                #print(xyz.shape)
                #raise Exception
                center = 0.5 * (np.max(xyz,axis=0) + np.min(xyz,axis=0))
                xyz -= center
                xyz *= 0.0025
                xyz_in_expand = np.tile(xyz,(int(batch_size),1,1))
                l = np.ones(xyz.shape[0])
                l_expand = np.tile(l,(int(batch_size),1))

                #print("XYZ_ex:", xyz_in_expand)
                xyz_in_expand = torch.tensor(xyz_in_expand, device="cuda", dtype=torch.float32)
                #res = model.run_cls(xyz_in_expand, l_expand, False) #TODO: MODEL und SO
                logits = model(xyz_in_expand)
                #print(logits.shape)
                #raise Exception
                fd1 = get_feature_dict_sep((xyz/0.0025)+center, np.asarray(logits.cpu().argmax(dim=1))[0], normals_r, tor)
                if norm_s in norm_fd.keys():
                    fdicts = norm_fd[norm_s]
                else:
                    norm_rr = np.around(norm_r)
                    # print norm_rr
                    norm_ss = ''
                    for i in norm_rr:
                        if i == 0:
                            norm_ss += str(0.0)
                        else:
                            norm_ss += str(i)
                    if norm_ss in norm_fd.keys():
                        fdicts = norm_fd[norm_ss]
                    else:
                        ran_key = random.sample(norm_fd.keys(), 1)
                        fdicts = norm_fd[ran_key[0]]

                mindiff = 10000
                matched_temp = ''
                matched_fd = {}
                for fdict in fdicts:
                    with open(fdict, 'rb') as tf:
                        fd2 = pickle.load(tf)
                    diff = similarity_sep(fd1, fd2)
                    if diff < mindiff:
                        mindiff = diff
                        matched_temp = os.path.split(fdict)[-1]
                        matched_fd = fd2
                # print 'min_diff: ', mindiff
                print('matched template: ', matched_temp)
                print('----------------------------------------')

                matched_name = os.path.splitext(matched_temp)[0]
                write_found_pose_in_sep(folder, namestr, frames[0], dict_all[matched_name],rot)
                match_dict[namestr] = matched_name
                end = time.time()
                t.append(end-start)
        with open(os.path.join('./data/test/results', folder, 'matched_dict.pkl'), 'wb') as tf:
            pickle.dump(match_dict,tf,protocol=2)
        print('Average look up time for one test')
        print(np.mean(np.array(t)))

@torch.no_grad()
def get_feature_dict_sep(xyz, label, normals, tor):
    '''Generate the feature dict for tested slice, the dict contains
    the nummber of each class and the coordinates of bounding box of each cluster of class'''
    feature_dict = {}
    elements = []
    feature_dict['normals'] = normals.astype(float)
    feature_dict['tor'] = tor
    min_edge = np.min(np.max(xyz, axis=0)-np.min(xyz, axis=0))
    for i in range(len(labeldict)):
        idx = np.argwhere(label==i)
        idx = np.squeeze(idx)
        #print(idx)
        #print(label.shape)
        #print(len(idx))
        l_i = label[idx]
        xyz_i = xyz[idx]
        if xyz_i.shape[0]>10:
            eps = min_edge / 2
            #print("Shape ", xyz_i.shape)
            c = DBSCAN(eps=eps, min_samples=10).fit(xyz_i)
            number = c.labels_.max()+1
            feature_info = np.zeros(shape = (number, 8, 3), dtype = float)
            for _ in range(number):
                idx_f = np.argwhere(c.labels_==_)
                idx_f = np.squeeze(idx_f)
                xyz_f = xyz_i[idx_f] # each cluster of each class
                l_f = l_i[idx_f]
                geometry = o3d.geometry.PointCloud()
                geometry.points = o3d.utility.Vector3dVector(xyz_f)
                bbox = geometry.get_axis_aligned_bounding_box()
                elements.append(geometry)
                elements.append(bbox)            
                feature_info[_,:,:] = np.asarray(bbox.get_box_points())
            feature_dict[labeldict[i]] = feature_info
        else:
            feature_dict[labeldict[i]] = None
    return feature_dict

@torch.no_grad()
def similarity_sep(feature_dict1, feature_dict2):
    '''Comparing the differences between two feature dictionaries using Euclidean distance
    '''
    loss_amount = 0
    loss_geo = 0
    loss_norm = 0
    norm1 = feature_dict1['normals']
    norm2 = feature_dict2['normals']
    loss_norm = np.sum((norm1-norm2)**2)
    tor1 = feature_dict1['tor']
    tor2 = feature_dict2['torch']
    loss_tor = int(tor1==tor2)
    for i in range(len(labeldict)):
        # print feature_dict1[labeldict[i]]
        # get the number of each class
        if type(feature_dict1[labeldict[i]]) == type(None):
            class_num_1_cur = 0
        else:
            class_num_1_cur = feature_dict1[labeldict[i]].shape[0]
            
        if type(feature_dict2[labeldict[i]]) == type(None):
            class_num_2_cur = 0
        else:
            class_num_2_cur = feature_dict2[labeldict[i]].shape[0]
        
        if class_num_1_cur == class_num_2_cur and class_num_1_cur != 0:
            f1 = feature_dict1[labeldict[i]]
            f1.sort(axis=0)
            f2 = feature_dict2[labeldict[i]]
            f2.sort(axis=0)
            for _ in range(class_num_1_cur):
                box1_all_points = f1[_] #shape(8, 3)
                box2_all_points = f2[_] #shape(8, 3)
                loss_geo += np.sum((box1_all_points-box2_all_points)**2)
        loss_amount += abs(class_num_1_cur-class_num_2_cur)
    return 10*loss_norm + 10*loss_tor + loss_amount + loss_geo/100000

@torch.no_grad()
def write_found_pose_in_sep(folder, filename, frame, pose, rot):
    '''Write the found poses to the xml file
    '''
    tor_dict = {'0': 'MRW510_CDD_10GH', '1': 'TAND_GERAD_DD'}
    
    doc = Document()  # create DOM
    FRAME_DUMP = doc.createElement('FRAME-DUMP') # create root element
    FRAME_DUMP.setAttribute('VERSION', '1.0') 
    FRAME_DUMP.setAttribute('Baugruppe', 'test')
    doc.appendChild(FRAME_DUMP)
    SNaht = doc.createElement('SNaht')
    SNaht.setAttribute('Name',frame[0])
    SNaht.setAttribute('ZRotLock',frame[1])
    SNaht.setAttribute('WkzName',frame[2])
    SNaht.setAttribute('WkzWkl',frame[3])
    FRAME_DUMP.appendChild(SNaht)

    Kontur = doc.createElement('Kontur')
    SNaht.appendChild(Kontur)

    Punkt = doc.createElement('Punkt')
    Punkt.setAttribute('X', frame[4])
    Punkt.setAttribute('Y', frame[5])
    Punkt.setAttribute('Z', frame[6])
    Kontur.appendChild(Punkt)

    Fl_Norm1 = doc.createElement('Fl_Norm')
    Fl_Norm1.setAttribute('X', frame[7])
    Fl_Norm1.setAttribute('Y', frame[8])
    Fl_Norm1.setAttribute('Z', frame[9])
    Punkt.appendChild(Fl_Norm1)

    Fl_Norm2 = doc.createElement('Fl_Norm')
    Fl_Norm2.setAttribute('X', frame[10])
    Fl_Norm2.setAttribute('Y', frame[11])
    Fl_Norm2.setAttribute('Z', frame[12])
    Punkt.appendChild(Fl_Norm2)
    
    Rot = doc.createElement('Rot')
    Rot.setAttribute('X', frame[13])
    Rot.setAttribute('Y', frame[14])
    Rot.setAttribute('Z', frame[15])
    Punkt.appendChild(Rot)
    EA = doc.createElement('Ext-Achswerte')
    EA.setAttribute('EA3', str(frame[16]))
    Punkt.appendChild(EA)

    Frames = doc.createElement('Frames')
    SNaht.appendChild(Frames)

    Frame = doc.createElement('Frame')
    Frames.appendChild(Frame)

    Pos = doc.createElement('Pos')
    Pos.setAttribute('X', frame[4])
    Pos.setAttribute('Y', frame[5])
    Pos.setAttribute('Z', frame[6])
    Frame.appendChild(Pos)
    
    rot_matrix = linalg.expm(np.cross(np.eye(3), [1,0,0] / linalg.norm([1,0,0]) * (-rot[0])))
    print(pose)
    xv = pose[-9:-6]

    xv_r = np.matmul(rot_matrix, xv.T)
    XVek = doc.createElement('XVek')
    XVek.setAttribute('X', str(xv_r[0]))
    XVek.setAttribute('Y', str(xv_r[1]))
    XVek.setAttribute('Z', str(xv_r[2]))
    Frame.appendChild(XVek)
    yv = pose[-6:-3]
    yv_r = np.matmul(rot_matrix, yv.T)
    YVek = doc.createElement('YVek')
    YVek.setAttribute('X', str(yv_r[0]))
    YVek.setAttribute('Y', str(yv_r[1]))
    YVek.setAttribute('Z', str(yv_r[2]))
    Frame.appendChild(YVek)
    zv = pose[-3:]
    zv_r = np.matmul(rot_matrix, zv.T)
    ZVek = doc.createElement('ZVek')
    ZVek.setAttribute('X', str(zv_r[0]))
    ZVek.setAttribute('Y', str(zv_r[1]))
    ZVek.setAttribute('Z', str(zv_r[2]))
    Frame.appendChild(ZVek)
    f = open('./data/test/results/'+folder+'/'+filename+'.xml','w')
    f.write(doc.toprettyxml(indent = '  '))
    f.close()


@torch.no_grad()
def test(model, data_list, cfg, num_votes=1):
    """using a part of original point cloud as input to save memory.
    Args:
        model (_type_): _description_
        test_loader (_type_): _description_
        cfg (_type_): _description_
        num_votes (int, optional): _description_. Defaults to 1.
    Returns:
        _type_: _description_
    """
    model.eval()  # set model to eval mode
    all_cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
    set_random_seed(0)
    cfg.visualize = cfg.get('visualize', False)
    if cfg.visualize:
        from openpoints.dataset.vis3d import write_obj
        cfg.vis_dir = os.path.join(cfg.run_dir, 'visualization')
        os.makedirs(cfg.vis_dir, exist_ok=True)
        cfg.cmap = cfg.cmap.astype(np.float32) / 255.

    # data
    trans_split = 'val' if cfg.datatransforms.get('test', None) is None else 'test'
    pipe_transform = build_transforms_from_cfg(trans_split, cfg.datatransforms)

    dataset_name = cfg.dataset.common.NAME.lower()
    len_data = len(data_list)

    cfg.save_path = cfg.get('save_path', f'results/{cfg.task_name}/{cfg.dataset.test.split}/{cfg.cfg_basename}')
    if 'semantickitti' in cfg.dataset.common.NAME.lower():
        cfg.save_path = os.path.join(cfg.save_path, str(cfg.dataset.test.test_id + 11), 'predictions')
    os.makedirs(cfg.save_path, exist_ok=True)

    gravity_dim = cfg.datatransforms.kwargs.gravity_dim
    nearest_neighbor = cfg.get('test_mode', 'multi_voxel') == 'nearest_neighbor'
    for cloud_idx, data_path in enumerate(data_list):
        logging.info(f'Test [{cloud_idx}]/[{len_data}] cloud')
        cm = ConfusionMatrix(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)
        all_logits = []
        coord, feat, label, idx_points, voxel_idx, reverse_idx_part, reverse_idx  = load_data(data_path, cfg)
        if label is not None:
            label = torch.from_numpy(label.astype(np.int).squeeze()).cuda(non_blocking=True)

        len_part = len(idx_points)
        nearest_neighbor = len_part == 1
        pbar = tqdm(range(len(idx_points)))
        for idx_subcloud in pbar:
            pbar.set_description(f"Test on {cloud_idx}-th cloud [{idx_subcloud}]/[{len_part}]]")
            if not (nearest_neighbor and idx_subcloud>0):
                idx_part = idx_points[idx_subcloud]
                coord_part = coord[idx_part]
                coord_part -= coord_part.min(0)

                feat_part =  feat[idx_part] if feat is not None else None
                data = {'pos': coord_part}
                if feat_part is not None:
                    data['x'] = feat_part
                if pipe_transform is not None:
                    data = pipe_transform(data)
                if 'heights' in cfg.feature_keys and 'heights' not in data.keys():
                    if 'semantickitti' in cfg.dataset.common.NAME.lower():
                        data['heights'] = torch.from_numpy((coord_part[:, gravity_dim:gravity_dim + 1] - coord_part[:, gravity_dim:gravity_dim + 1].min()).astype(np.float32)).unsqueeze(0)
                    else:
                        data['heights'] = torch.from_numpy(coord_part[:, gravity_dim:gravity_dim + 1].astype(np.float32)).unsqueeze(0)
                if not cfg.dataset.common.get('variable', False):
                    if 'x' in data.keys():
                        data['x'] = data['x'].unsqueeze(0)
                    data['pos'] = data['pos'].unsqueeze(0)
                else:
                    data['o'] = torch.IntTensor([len(coord)])
                    data['batch'] = torch.LongTensor([0] * len(coord))

                for key in data.keys():
                    data[key] = data[key].cuda(non_blocking=True)
                data['x'] = get_features_by_keys(data, cfg.feature_keys)
                logits = model(data)
                """visualization in debug mode. !!! visulization is not correct, should remove ignored idx.
                from openpoints.dataset.vis3d import vis_points, vis_multi_points
                vis_multi_points([coord, coord_part], labels=[label.cpu().numpy(), logits.argmax(dim=1).squeeze().cpu().numpy()])
                """

            all_logits.append(logits)
        all_logits = torch.cat(all_logits, dim=0)
        if not cfg.dataset.common.get('variable', False):
            all_logits = all_logits.transpose(1, 2).reshape(-1, cfg.num_classes)

        if not nearest_neighbor:
            # average merge overlapped multi voxels logits to original point set
            idx_points = torch.from_numpy(np.hstack(idx_points)).cuda(non_blocking=True)
            all_logits = scatter(all_logits, idx_points, dim=0, reduce='mean')
        else:
            # interpolate logits by nearest neighbor
            all_logits = all_logits[reverse_idx_part][voxel_idx][reverse_idx]
        pred = all_logits.argmax(dim=1)
        if label is not None:
            cm.update(pred, label)
        """visualization in debug mode
        from openpoints.dataset.vis3d import vis_points, vis_multi_points
        vis_multi_points([coord, coord], labels=[label.cpu().numpy(), all_logits.argmax(dim=1).squeeze().cpu().numpy()])
        """
        if cfg.visualize:
            gt = label.cpu().numpy().squeeze() if label is not None else None
            pred = pred.cpu().numpy().squeeze()
            gt = cfg.cmap[gt, :] if gt is not None else None
            pred = cfg.cmap[pred, :]
            # output pred labels
            if 's3dis' in dataset_name:
                file_name = f'{dataset_name}-Area{cfg.dataset.common.test_area}-{cloud_idx}'
            else:
                file_name = f'{dataset_name}-{cloud_idx}'

            write_obj(coord, feat,
                      os.path.join(cfg.vis_dir, f'input-{file_name}.obj'))
            # output ground truth labels
            if gt is not None:
                write_obj(coord, gt,
                        os.path.join(cfg.vis_dir, f'gt-{file_name}.obj'))
            # output pred labels
            write_obj(coord, pred,
                      os.path.join(cfg.vis_dir, f'{cfg.cfg_basename}-{file_name}.obj'))

        if cfg.get('save_pred', False):
            if 'semantickitti' in cfg.dataset.common.NAME.lower():
                pred = pred + 1
                pred = pred.cpu().numpy().squeeze()
                pred = pred.astype(np.uint32)
                upper_half = pred >> 16  # get upper half for instances
                lower_half = pred & 0xFFFF  # get lower half for semantics (lower_half.shape) (100k+, )
                lower_half = remap_lut_write[lower_half]  # do the remapping of semantics
                pred = (upper_half << 16) + lower_half  # reconstruct full label
                pred = pred.astype(np.uint32)
                frame_id = data_path[0].split('/')[-1][:-4]
                store_path = os.path.join(cfg.save_path, frame_id + '.label')
                pred.tofile(store_path)
            elif 'scannet' in cfg.dataset.common.NAME.lower():
                pred = pred.cpu().numpy().squeeze()
                label_int_mapping={0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 14, 13: 16, 14: 24, 15: 28, 16: 33, 17: 34, 18: 36, 19: 39}
                pred=np.vectorize(label_int_mapping.get)(pred)
                save_file_name=data_path.split('/')[-1].split('_')
                save_file_name=save_file_name[0]+'_'+save_file_name[1]+'.txt'
                save_file_name=os.path.join(cfg.save_path,save_file_name)
                np.savetxt(save_file_name, pred, fmt="%d")

        if label is not None:
            tp, union, count = cm.tp, cm.union, cm.count
            miou, macc, oa, ious, accs = get_mious(tp, union, count)
            with np.printoptions(precision=2, suppress=True):
                logging.info(
                    f'[{cloud_idx}]/[{len_data}] cloud,  test_oa , test_macc, test_miou: {oa:.2f} {macc:.2f} {miou:.2f}, '
                    f'\niou per cls is: {ious}')
            all_cm.value += cm.value

    if 'scannet' in cfg.dataset.common.NAME.lower():
        logging.info(f" Please select and zip all the files (DON'T INCLUDE THE FOLDER) in {cfg.save_path} and submit it to"
                     f" Scannet Benchmark https://kaldir.vc.in.tum.de/scannet_benchmark/. ")

    if label is not None:
        tp, union, count = all_cm.tp, all_cm.union, all_cm.count
        if cfg.distributed:
            dist.all_reduce(tp), dist.all_reduce(union), dist.all_reduce(count)
        miou, macc, oa, ious, accs = get_mious(tp, union, count)
        return miou, macc, oa, ious, accs, all_cm
    else:
        return None, None, None, None, None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.cfg_basename,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path

    # wandb config
    cfg.wandb.name = cfg.run_name

    # multi processing.
    #if cfg.mp:
     #   port = find_free_port()
      #  cfg.dist_url = f"tcp://localhost:{port}"
       # print('using mp spawn for distributed training')
        #mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
    #else:
    main(0, cfg)