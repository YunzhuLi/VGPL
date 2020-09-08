CUDA_VISIBLE_DEVICES=0		\
python eval_pos.py		\
	--env RigidFall 	\
	--stage pos		\
	--eval_epoch 78		\
	--eval_iter 4000	\
	--n_his 4		\
	--sequence_length 20	\
	--augment 0.05		\
	--discard_frames 0	\
	--vispy 0		\
