#!/bin/sh


# This script runs a Hugging Face model as an API endpoint via TGI.
# This script is invoked in autollm.sh, and should be run from there e.g.
# ssh -qt USER@e2.tch.harvard.edu /path/to/autollm.sh llama3-70b-eetq


## ** Start of script **

WORKDIR=$(realpath $(dirname $0))
HOMEDIR=$HOME/autollm

mkdir -p $HOMEDIR

echo
echo "Script running on $(hostname -s)."

if [ -z "$MODEL_ARGS" ]; then
  echo "No model args - did you run this directly? Try running llama3-70b.sh instead."
  exit 1
fi

# Download container if needed
TGI_VERSION="2.0.4"
TGI_IMAGE="$WORKDIR/text-generation-inference_$TGI_VERSION.sif"
if [ ! -f $TGI_IMAGE ]; then
  echo "Downloading Hugging Face wrapper (first run only, this takes approx 45min)"
  mkdir -p /temp_work/$USER/autollm
  TMPDIR=/temp_work/$USER/autollm apptainer pull $TGI_IMAGE docker://ghcr.io/huggingface/text-generation-inference:$TGI_VERSION
fi

# Launch LLM
echo "Launching model (may take a while)"

# NOTE: For gated models, you need to specify your HF token here
export HUGGING_FACE_HUB_TOKEN=<TOKEN>

mkdir -p $WORKDIR/tgi_data

# Find open port, starting your search based on $1 
get_unused_port() {
  for port in $(seq $1 65000); do
    if ! netstat -lnt | grep ":$port " >/dev/null; then
      echo $port
      break
    fi
  done
}

APPTAINER_PORT=8888

# Give some space between SHARD port and APPTAINER port 
SHARD_PORT=$(get_unused_port 9000)
# Get a temp folder for this container run
SCRATCH_SPACE=$(mktemp -d)

# Run the model
export RUST_BACKTRACE=full # helps debugging
# Also when debugging: remove the /dev/null piping for better ShardError info
# apptainer run --nv --bind $WORKDIR/tgi_data:/data $TGI_IMAGE --port $APPTAINER_PORT --master-port $SHARD_PORT --shard-uds-path $SCRATCH_SPACE/tgi $MODEL_ARGS >/dev/null &
echo "apptainer run --nv --bind $WORKDIR/tgi_data:/data $TGI_IMAGE --port $APPTAINER_PORT --master-port $SHARD_PORT --shard-uds-path $SCRATCH_SPACE/tgi $MODEL_ARGS  &"
apptainer run --nv --bind $WORKDIR/tgi_data:/data $TGI_IMAGE --port $APPTAINER_PORT --master-port $SHARD_PORT --shard-uds-path $SCRATCH_SPACE/tgi $MODEL_ARGS  &
APPTAINER_PID=$!

# Wait for startup
while ! curl -s "localhost:$APPTAINER_PORT"; do
  sleep 1
done

# Done - Pass back to the user
HOSTNAME=$(hostname -I | cut -d' ' -f1)  # get first ip
echo
echo "All set!"
echo
echo "Run the following command in a separate terminal:"
echo "ssh -qN -L 8888:$HOSTNAME:$APPTAINER_PORT $USER@e2.tch.harvard.edu"
echo
echo "Then make POST requests to following URL:"
echo "http://localhost:8888"
echo
echo "Press Ctrl+C in this terminal when you are all done."

sleep infinity
