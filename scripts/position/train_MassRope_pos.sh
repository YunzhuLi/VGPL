CUDA_VISIBLE_DEVICES=0		\
python train.py 		\
	--env MassRope 		\
	--stage pos 	   	\
	--gen_data 0 		\
	--gen_stat 0		\
	--gen_vision 0		\
	--num_workers 10 	\
	--resume 0          	\
	--resume_epoch 85    	\
	--resume_iter 110000 	\
	--lr 1e-5		\
	--optimizer Adam 	\
	--batch_size 2		\
	--n_his 4		\
	--sequence_length 10	\
	--augment 0.05   	\
	--discard_frames 20	\
	--verbose_data 0 	\
	--verbose_model 0	\
	--log_per_iter 20	\
	--ckp_per_iter 2000	\
	--eval 0
	
