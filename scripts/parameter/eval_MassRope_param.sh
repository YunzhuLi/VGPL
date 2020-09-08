CUDA_VISIBLE_DEVICES=0		\
python eval_param.py		\
	--env MassRope 		\
	--stage param		\
	--eval_epoch 89		\
	--eval_iter 30000	\
	--n_his 4		\
	--sequence_length 20	\
	--augment 0.05		\
	--discard_frames 20	\
