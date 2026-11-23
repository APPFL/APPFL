#!/bin/bash
#SBATCH --job-name=dl-evo2-7b-base
#SBATCH --account=m4994
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft/logs/download_evo2_7b_base.out
#SBATCH --error=/pscratch/sd/a/abhijit7/dbforge/wastewater-peft/logs/download_evo2_7b_base.err

# One-time download of the 8K Evo2 base model used by client_perlmutter.yaml.
# Submit from a Perlmutter login node: sbatch download_evo2_7b_base.sh

set -euo pipefail

SHARED_ENV=/global/cfs/projectdirs/m4994/common/db-forge-poc/evo2_metagenomics/envs/evo2_fl_env312
DEST=/global/cfs/projectdirs/m4994/common/db-forge-poc/models/evo2_7b_base

module load conda
conda activate "$SHARED_ENV"
mkdir -p "$DEST/hub"

SNAP=$(python - <<EOF
from huggingface_hub import snapshot_download
print(snapshot_download("arcinstitute/evo2_7b_base", cache_dir="$DEST/hub"))
EOF
)
echo "SNAPSHOT $SNAP"
ls -laL "$SNAP"

# single file or .partN shards (merged in order, same as evo2's loader)
if [ -f "$SNAP/evo2_7b_base.pt" ]; then
    ln -sfn "$SNAP/evo2_7b_base.pt" "$DEST/evo2_7b_base.pt"
else
    : > "$DEST/evo2_7b_base.pt"
    i=0
    while [ -f "$SNAP/evo2_7b_base.pt.part$i" ]; do
        cat "$SNAP/evo2_7b_base.pt.part$i" >> "$DEST/evo2_7b_base.pt"
        i=$((i + 1))
    done
    echo "merged $i shards"
fi

ls -laL "$DEST/evo2_7b_base.pt"
sha256sum "$DEST/evo2_7b_base.pt"
chmod -R g+rX "$DEST"
