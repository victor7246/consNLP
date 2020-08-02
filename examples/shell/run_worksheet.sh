### CodaLab arguments
CODALAB_ARGS="cl run"

# Name of bundle (can customize however you want)
CODALAB_ARGS="$CODALAB_ARGS --name run-train"
# Docker image (default: codalab/default-cpu)
CODALAB_ARGS="$CODALAB_ARGS --request-docker-image datamafia7/codalab-python:latest"
# Explicitly ask for a worker with at least one GPU
CODALAB_ARGS="$CODALAB_ARGS --request-gpus 1"
# Control the amount of RAM your run needs
CODALAB_ARGS="$CODALAB_ARGS --request-memory 10g"
# Kill job after this many days (default: 1 day)
CODALAB_ARGS="$CODALAB_ARGS --request-time 2d"

# Bundle dependencies
CODALAB_ARGS="$CODALAB_ARGS :src"                              # Code
CODALAB_ARGS="$CODALAB_ARGS :data"                             # Dataset
CODALAB_ARGS="$CODALAB_ARGS :models"  # models

### Command to execute (these flags can be overridden) from the command-line
CMD="python3 src/main.py"
# Read in the dataset
CMD="$CMD --train_data data/raw/COVID19Tweet-master/train.tsv"
CMD="$CMD --val_data data/raw/COVID19Tweet-master/valid.tsv"
CMD="$CMD --transformer_model_pretrained_path roberta-base"
CMD="$CMD --bpe_vocab_path models/vocab.json"
CMD="$CMD --bpe_merges_path models/merges.txt"
CMD="$CMD --max_text_len 300"
CMD="$CMD --epochs 5"
CMD="$CMD --lr .00002"
CMD="$CMD --loss_function bcelogit"
CMD="$CMD --metric f1"
CMD="$CMD --use_torch_trainer True"
CMD="$CMD --use_gpu True"
CMD="$CMD --train_batch_size 16"
CMD="$CMD --eval_batch_size 32"
CMD="$CMD --model_path /"
CMD="$CMD --seed 42"
# Pass the command-line arguments through to override the above
if [ -n "$1" ]; then
  CMD="$CMD $@"
fi

# Create the run on CodaLab!
FINAL_COMMAND="$CODALAB_ARGS '$CMD'"
echo $FINAL_COMMAND
exec bash -c "$FINAL_COMMAND"