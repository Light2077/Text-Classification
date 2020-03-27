export BERT_BASE_DIR=pretrained_model/chinese_L-12_H-768_A-12
export DATA_DIR=../data

python run_classifier.py \
  --task_name=BAIDU \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=tmp/baidu_output/ \
  --max_seq_length=128 \
  --output_dir=tmp/baidu_output/

echo "press any key to continue"
read