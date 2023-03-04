# !/bin/bash
export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

CURR_DBSCAN=0.95
CURR_TOPK=750
CURR_QUERY=150

# # TRAIN
python main_instance_segmentation.py \
general.experiment_name="Mask3D_OW" \
general.project_name="open_world_instance_segmentation" \
general.train_mode=true \
general.checkpoint=null \
data/datasets=scannet200 \
general.train_oracle=False \
general.learn_energy_trainig_dataset=False \
general.enable_baseline_clustering=False \
general.clustering_start_iter=20000 \
general.clustering_update_mu_iter=20 \
general.clustering_momentum=0.7
# # data.num_labels=200 \
# # general.eval_on_segments=true \
# # general.train_on_segments=true

# TEST
#python main_instance_segmentation.py \
#general.experiment_name="test" \ 
#general.project_name="test" \
#general.checkpoint="/l/users/mohamed.boudjoghra/Research/Mask3D/saved/scannet200_head_OW/best.ckpt" \
#data/datasets=scannet200 \
#general.num_targets=202 \
#data.num_labels=201 \
#general.eval_on_segments=true \
#general.train_on_segments=true \
#general.train_mode=false \
##model.num_queries=150 \
#general.topk_per_image=500 \
# general.use_dbscan=true \
# general.dbscan_eps=0.95
