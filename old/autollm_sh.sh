#!/bin/sh
# call via ./autollm.sh llama3-8b-test
WORKDIR="$(realpath $(dirname $0))/autollm"
MODEL_ARGS=""

if [ -z "$1" ]; then
  echo "Model name required (try llama3-70b)"
  exit 1
fi
MODEL_NAME="$1"

# https://stackoverflow.com/a/18558871/239668
beginswith() { case $1 in "$2"*) true;; *) false;; esac; }

# ** Llama 2 and Llama3  **
if beginswith "$MODEL_NAME" llama; then
  if [ "$MODEL_NAME" = "llama3-70b-eetq" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id meta-llama/Meta-Llama-3-70B-Instruct"
    MODEL_ARGS="$MODEL_ARGS --revision 359ec69a0f92259a3cd2da3bb01d31e16c260cfc"
    # Should double the context window as a baseline
    # See https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md 
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=16000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=16000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=18000"
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    # DIDN'T WORK :'(
    # – Disabling CUDA graphs to try and fix some weird token issues? 　
    # https://github.com/huggingface/text-generation-inference/issues/1723　# MODEL_ARGS="$MODEL_ARGS --cuda-graphs 0"
    MODEL_ARGS="$MODEL_ARGS --rope-scaling dynamic"  # this lets us do >4k tokens for llama2 and 8K for llama3
    # MODEL_ARGS="$MODEL_ARGS --rope-factor 2.0"
    MODEL_ARGS="$MODEL_ARGS --num-shard=2"
    SLURM_ARGS="--gres=gpu:NVIDIA_A100:2 --mem 32G"
  elif [ "$MODEL_NAME" = "llama3-8b" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_ARGS="$MODEL_ARGS --revision c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
    # Should double the context window as a baseline
    # See https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md 
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=8000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=8000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=10000"
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    MODEL_ARGS="$MODEL_ARGS --rope-scaling dynamic"  # this lets us do >4k tokens for llama2 and 8K for llama3
    SLURM_ARGS="--gres=gpu:NVIDIA_A100:1 --mem 32G"
  elif [ "$MODEL_NAME" = "llama3-8b-A40s" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_ARGS="$MODEL_ARGS --revision c4a54320a52ed5f88b7a2f84496903ea4ff07b45"
    # Should double the context window as a baseline
    # See https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md 
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=8000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=8000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=10000"
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    MODEL_ARGS="$MODEL_ARGS --rope-scaling dynamic"  # this lets us do >4k tokens for llama2 and 8K for llama3
    MODEL_ARGS="$MODEL_ARGS --num-shard=2"
    SLURM_ARGS="--gres=gpu:NVIDIA_A40:2 --nodelist=gpu-4-0 --mem 32G"
  elif [ "$MODEL_NAME" = "llama2-70b-eetq" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id meta-llama/Llama-2-70b-chat-hf"
    MODEL_ARGS="$MODEL_ARGS --revision 36d9a7388cc80e5f4b3e9701ca2f250d21a96c30"
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=9000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=9000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=10000"
    MODEL_ARGS="$MODEL_ARGS --rope-scaling dynamic"  # this lets us do >4k tokens for llama2 and 8K for llama3
    MODEL_ARGS="$MODEL_ARGS --rope-factor 2.0"
    SLURM_ARGS="--gres=gpu:NVIDIA_A100:1 --mem 32G"
  fi


###############
# ** Mistral **
elif beginswith "$MODEL_NAME" mistral; then
  if [ "$MODEL_NAME" = "mistral-8x7b-eetq" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id mistralai/Mixtral-8x7B-Instruct-v0.1"
    MODEL_ARGS="$MODEL_ARGS --revision 125c431e2ff41a156b9f9076f744d2f35dd6e67a"
    # See above for why we've chosen this length
    # Context length by default is 32k 
    # https://mistral.ai/news/mixtral-of-experts/
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=9000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=9000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=10000"
    # Shouldn't need rope scaling
    # MODEL_ARGS="$MODEL_ARGS --rope-scaling dynamic"  # this lets us do >4k tokens
    # MODEL_ARGS="$MODEL_ARGS --rope-factor 2.0"
    # Use eetq quantizing by default
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    SLURM_ARGS="--gres=gpu:NVIDIA_A100:1 --mem 32G"
  elif [ "$MODEL_NAME" = "mistral-8x22b-eetq" ]; then
    MODEL_ARGS="$MODEL_ARGS --model-id mistralai/Mixtral-8x22B-Instruct-v0.1"
    MODEL_ARGS="$MODEL_ARGS --revision 95d063951382d47385fe7b36e202b68639e5c066"
    # Should have a 64k context window https://mistral.ai/news/mixtral-8x22b/; limiting it for space
    # No need for rope scaling becuase of the default context window's size
    MODEL_ARGS="$MODEL_ARGS --max-input-tokens=32000"
    MODEL_ARGS="$MODEL_ARGS --max-batch-prefill-tokens=32000"
    MODEL_ARGS="$MODEL_ARGS --max-total-tokens=34000"
    # Use eetq quantizing by default
    MODEL_ARGS="$MODEL_ARGS --quantize eetq"
    MODEL_ARGS="$MODEL_ARGS --num-shard=2"
    SLURM_ARGS="--gres=gpu:NVIDIA_A100:2 --mem 32G"
  fi
fi

if [ -z "$MODEL_ARGS" ]; then
  echo "Unrecognized model name - try llama3-70b-eetq or mistral-8x22b-eetq "
  exit 1
fi


SCRIPT="run-llm.sh"

echo "Requesting $SLURM_ARGS..."
echo srun -A bch -p bch-gpu-pe --pty --job-name eliAutollm $SLURM_ARGS \
    env MODEL_ARGS="$MODEL_ARGS" $WORKDIR/$SCRIPT

exec srun -A bch -p bch-gpu-pe --pty --job-name eliAutollm $SLURM_ARGS \
    env MODEL_ARGS="$MODEL_ARGS" $WORKDIR/$SCRIPT
