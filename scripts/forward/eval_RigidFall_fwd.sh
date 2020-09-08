CUDA_VISIBLE_DEVICES=0		\
python eval_fwd.py		\
	--env RigidFall 	\
	--stage forward		\
	--eval_epoch_pos 78	\
	--eval_iter_pos 4000	\
	--eval_epoch_param 79	\
	--eval_iter_param 18000 \
	--n_his 4		\
	--sequence_length 30	\
	--augment 0.05		\
	--discard_frames 0	\
	--vispy 0		\
