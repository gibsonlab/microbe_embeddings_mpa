#!/bin/bash

# Change this as necessary.
processed_dir=/media/youn/data/projects/mpa_data/processed
outfile=$processed_dir/all_markers.fna.bgz

zstdcat $processed_dir/*.zst | bgzip -c > $outfile
