import logging
import os
from hashlib import md5
from uuid import uuid4
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from trainer.trainer import InstanceSegmentation, RegularCheckpointing
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import (
    flatten_dict,
    load_baseline_model, 
    load_checkpoint_with_missing_or_exsessive_keys,
    load_backbone_checkpoint_with_missing_or_exsessive_keys
)
from pytorch_lightning import Trainer, seed_everything
from omegaconf import OmegaConf,open_dict
from tqdm import tqdm
import json 
import torch




####################################################################################################################################
import numpy as np
from datasets.scannet200.scannet200_constants import CLASS_LABELS_200
from datasets.scannet200.scannet200_splits import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
####################################################################################################################################
############################
CLASS_LABELS_200 = list(CLASS_LABELS_200)
CLASS_LABELS_200.remove('floor')
CLASS_LABELS_200.remove('wall')
# HEAD_CATS_SCANNET_200.remove('floor')
# HEAD_CATS_SCANNET_200.remove('wall')
MAP_STRING_TO_ID = {CLASS_LABELS_200[i] : i for i in range(len(CLASS_LABELS_200))}
MAP_STRING_TO_ID['background'] = 253
############################

def save_recall_at(th_list=[0.25], path2file="./recall.yaml"):
        thresholds = th_list
        recall_at = {th : {} for th in thresholds}
        cuda0 = torch.device('cuda:0')
        for th in thresholds:
            for label in tqdm(HEAD_CATS_SCANNET_200+COMMON_CATS_SCANNET_200+TAIL_CATS_SCANNET_200):
                label_id = torch.Tensor([MAP_STRING_TO_ID[label]]).to(device=cuda0)
                mask = (torch.Tensor(cfg.gt_iou['gt_id']).to(device=cuda0)==label_id).to(device=cuda0)
                gt_count= torch.count_nonzero(mask)
                tp = torch.count_nonzero(torch.any((torch.Tensor(cfg.gt_iou['iou']).to(device=cuda0))[mask]>th, axis = 1))
                if gt_count:
                    r_per_class = tp.item()/gt_count.item()
                else:
                    r_per_class = 0
                    
                recall_at[th][label] = r_per_class
        
        with open(path2file, 'w') as fp:
            json.dump(recall_at, fp,  indent=4) 
def get_parameters(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    load_dotenv(".env")

    # parsing input parameters
    seed_everything(cfg.general.seed)

    # getting basic configuration 
    if cfg.general.get("gpus", None) is None:
        cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    loggers = []

    # cfg.general.experiment_id = "0" # str(Repo("./").commit())[:8]
    # params = flatten_dict(OmegaConf.to_container(cfg, resolve=True))

    # create unique id for experiments that are run locally
    # unique_id = "_" + str(uuid4())[:4]
    # cfg.general.version = md5(str(params).encode("utf-8")).hexdigest()[:8] + unique_id

    if not os.path.exists(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
    else:
        print("EXPERIMENT ALREADY EXIST")
        cfg['trainer']['resume_from_checkpoint'] = f"{cfg.general.save_dir}/last-epoch.ckpt"

    for log in cfg.logging:
        print(log)
        loggers.append(hydra.utils.instantiate(log))
        loggers[-1].log_hyperparams(
            flatten_dict(OmegaConf.to_container(cfg, resolve=True))
        )

    model = InstanceSegmentation(cfg)
    if cfg.general.backbone_checkpoint is not None:
        cfg, model = load_backbone_checkpoint_with_missing_or_exsessive_keys(cfg, model)
    if cfg.general.checkpoint is not None:
        cfg, model = load_checkpoint_with_missing_or_exsessive_keys(cfg, model)

    logger.info(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    return cfg, model, loggers


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def train(cfg: DictConfig):
    
    if cfg.general.save_recall:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.gt_iou = {"gt_id":[],"gt_label":[], "iou":[]}
            
    if True:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.A_OSE = 0
    
    # if cfg.general.save_KN_UKN_distance:
    #     OmegaConf.set_struct(cfg, True)
    #     with open_dict(cfg):
    #         cfg.save_KN_UKN_distance = {"gt_id":[],"gt_label":[], "features":[]}
        
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    callbacks = [] 
    for cb in cfg.callbacks:
        callbacks.append(hydra.utils.instantiate(cb))

    callbacks.append(RegularCheckpointing())

    runner = Trainer(
        logger=loggers,
        gpus=cfg.general.gpus,
        callbacks=callbacks,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.fit(model)


@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def test(cfg: DictConfig):
    # because hydra wants to change dir for some reason
    
    if cfg.general.save_recall:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.gt_iou = {"gt_id":[],"gt_label":[], "iou":[]}

    if True:
        OmegaConf.set_struct(cfg, True)
        with open_dict(cfg):
            cfg.A_OSE = 0
    
    # if cfg.general.save_KN_UKN_distance:
    #     OmegaConf.set_struct(cfg, True)
    #     with open_dict(cfg):
    #         cfg.querie_feats = {"pred_id":[],"pred_label":[], "quer_features":[]}

    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer
    )
    runner.test(model)

@hydra.main(config_path="conf", config_name="config_base_instance_segmentation.yaml")
def main(cfg: DictConfig):
    if cfg['general']['train_mode']:
        train(cfg)
    else:
        test(cfg)
        if cfg.general.save_recall:
            save_recall_at(th_list=cfg.general.save_recall_at, path2file=cfg.general.save_recall_in)
        

if __name__ == "__main__":    main()

