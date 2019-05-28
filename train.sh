python -m torch.distributed.launch \
	--nproc_per_node=2  \
	--master_port=$((RANDOM + 10000)) \
	tools/train.py \
	configs/retinanet_r50_fpn_1x.py \
	--launche pytorch \
	--validate 
	#--resume_from='work_dirs/retinanet_r50_fpn_1x/latest.pth'

