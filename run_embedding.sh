#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 14
#SBATCH -t 24:00:00

# model_name = args.model_name
# use_narrations = args.use_narrations
# layer_sizes = args.layer_sizes
# N_EPOCHS = args.n_epochs
# BATCH_SIZE = args.batch_size

# Default values
layer_sizes=None
use_narrations="False"
model_name=None
n_epochs=100
batch_size=64
min_sensible=0.0
min_correct=0.0

# Parse arguments
for i in "$@"
do
case $i in
    --layer_sizes=*)
    layer_sizes="${i#*=}"
    shift # past argument=value
    ;;
    --use_narrations)
    use_narrations="True"
    shift # past argument with no value
    ;;
    --model_name=*)
    model_name="${i#*=}"
    shift # past argument with no value
    ;;
    --batch_size=*)
    batch_size="${i#*=}"
    shift # past argument with no value
    ;;
    --n_epochs=*)
    n_epochs="${i#*=}"
    shift # past argument with no value
    ;;
    --min_sensible=*)
    min_sensible="${i#*=}"
    shift # past argument with no value
    ;;
    --min_correct=*)
    min_correct="${i#*=}"
    shift # past argument with no value
    ;;
    *)
    # unknown option
    ;;
esac
done

# Prepare optional parameter strings
args=""
for var in layer_sizes use_narrations model_name batch_size n_epochs min_sensible min_correct; do
    value=$(eval echo \$$var)
    if [ "$value" != "None" ]; then
        args+=" --$var=$value"
    fi
done

echo "Running with args: $args"

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source ./venv/bin/activate

python train_embedding_model.py $args