python -m torch.distributed.launch \
	--nproc_per_node=2  \
	--master_port=$((RANDOM + 10000)) \
	tools/train.py \
	--launche pytorch \
	--validate  \
	# configs/retinanet_r50_fpn_1x_bdd.py 
	configs/retinanet_efficientnet_b3_fpn_1x_bdd.py

