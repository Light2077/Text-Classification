export CKPT_DIR='../notebook/TextCNN/checkpoints/train'

python -u train.py \
		--max_len=128 \
		--embedding_dim=256 \
		--filters=2 \
		--kernel_sizes='2,3,4' \
		--batch_size=256 \
		--dropout_rate=0.1 \
		--epochs=1 \
		--learning_rate=0.01 \
		--checkpoint_path=$CKPT_DIR

echo "press any key to continue"
read