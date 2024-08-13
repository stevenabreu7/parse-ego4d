#!/bin/bash
#SBATCH -p g48
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source ./venv/bin/activate

python gpt_0shot.py