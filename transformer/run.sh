export CKPT_DIR='../notebook/Transformer/checkpoints/train'

python -u train.py \
		--num_layers=4 \
		--d_model=128 \
		--num_heads=8 \
		--dff=512 \
		--batch_size=128 \
		--max_position_encoding=10000 \
		--output_dim=97 \
		--dropout_rate=0.1 \
		--epochs=1 \
		--learning_rate=0.001 \
		--checkpoint_path=$CKPT_DIR

echo "press any key to continue"
read