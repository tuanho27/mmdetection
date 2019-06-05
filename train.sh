python -m torch.distributed.launch \
	--nproc_per_node=2  \
	--master_port=$((RANDOM + 10000)) \
	tools/train.py \
	configs/retinanet_r50_fpn_1x_bdd.py \
	--launche pytorch \
	--validate  

