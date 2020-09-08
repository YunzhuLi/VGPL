CUDA_VISIBLE_DEVICES=0		\
python eval_pos.py		\
	--env MassRope 		\
	--stage pos		\
	--eval_epoch 86		\
	--eval_iter 160000	\
	--n_his 4		\
	--sequence_length 20	\
	--augment 0.05		\
	--discard_frames 20	\
	--vispy 0		\
