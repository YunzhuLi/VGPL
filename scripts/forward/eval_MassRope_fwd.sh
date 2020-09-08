CUDA_VISIBLE_DEVICES=0		\
python eval_fwd.py		\
	--env MassRope 		\
	--stage forward		\
	--eval_epoch_pos 87	\
	--eval_iter_pos 132000	\
	--eval_epoch_param 89	\
	--eval_iter_param 30000	\
	--n_his 4		\
	--sequence_length 30	\
	--augment 0.05		\
	--discard_frames 20	\
	--vispy 1		\
