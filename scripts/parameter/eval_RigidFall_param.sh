CUDA_VISIBLE_DEVICES=0		\
python eval_param.py		\
	--env RigidFall 	\
	--stage param		\
	--eval_epoch 79		\
	--eval_iter 18000	\
	--n_his 4		\
	--sequence_length 20	\
	--augment 0.05		\
	--discard_frames 0	\
