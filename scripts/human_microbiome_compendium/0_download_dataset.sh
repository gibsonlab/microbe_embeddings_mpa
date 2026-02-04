#!/bin/bash
set -e

requires_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: '$1' is required but not installed." >&2
        echo "Please install it using your package manager." >&2
        exit 1
    fi
}
requires_command zstd
requires_command gzip

# Script for downloading and preprocessing the human_microbiome_compendium dataset.
TARGET_DIR="/data/cctm/youn/human_microbiome_compendium"
echo "TARGET_DIR: ${TARGET_DIR}"
echo "The rest of the script assumes the above directory stores all of the extracted files. "
echo "If this is not where the file is located, please change this directory in the rest of the pipeline, for each of the scripts separately."


if [ -d "${TARGET_DIR}" ]; then
  echo "Directory ${TARGET_DIR} found. Proceeding preprocessing workflow..."
  cd ${TARGET_DIR}
else
  exit 1
fi


echo "(Step 1 of 3) Fetching v1.1.1 of the Human Microbiome compendium from Zenodo..."
curl "https://zenodo.org/api/records/15122187/files-archive" -o "./archive.zip"
unzip ./archive.zip
rm archive.zip


echo "(Step 2 of 3) Repacking ASV sequence file... (.gz -> .zstd)"
gunzip obs_md.txt.gz
zstd --rm obs_md.txt


echo "(Step 3 of 3) Unpacking ASV count files..."
tar -xzf project_asv_tables.tar.gz  # this unpacks a subdir called "asv"
cd asv
echo "To reduce disk footprint, this script will now compress each ASV count file."
for counts_file in ./*.txt; do
  echo "Compressing: ${counts_file}"
  zstd --rm "${counts_file}"
done

echo ""
echo "Done!"
