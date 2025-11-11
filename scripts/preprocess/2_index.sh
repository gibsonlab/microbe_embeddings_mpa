#!/bin/bash

# Change this as necessary.
processed_dir=/data/cctm/youn/metaphlan_dset/phylophlan_data/processed
outfile=$processed_dir/all_markers.fna.bgz

zstdcat $processed_dir/*.zst | bgzip -c > $outfile
