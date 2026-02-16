"""
Step 1: Take the multiple alignment file output by PhyloPhlAn (amino acid alignments) and remove all gaps from each
sequence entry. This recovers the amino acid sequences of the gene markers.
"""
from pathlib import Path
import bz2
from typing import *

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def gather_markers(sgb_marker_file: Path, out_file: Path, seen_ids: Set[str]):
    print(f"Extracting markers from: {sgb_marker_file}")
    file_suffix = '.fna.bz2'
    assert sgb_marker_file.name.endswith(file_suffix), f"Expected sgb marker FASTA file to have extension {file_suffix}. Filename was: {sgb_marker_file.name}"

    with bz2.open(sgb_marker_file, "rt") as seq_f, open(out_file, "at") as out_f:
        for record in SeqIO.parse(seq_f, "fasta"):
            gene_name = record.id.split(":")[0].split("_")[-1]
            assert gene_name.startswith("p") and len(gene_name) == 5, f"Expected to find gene name of the form `p<4-digit-id>`, but got `{gene_name}` instead."
            sgb_genome_name = sgb_marker_file.name[:-len(file_suffix)]
            seq = record.seq
            new_record = SeqRecord(seq, id=f"{gene_name}:{sgb_genome_name}", description="")
            if new_record.id in seen_ids:
                print(f"[WARNING] {new_record.id} already in use. Skipping!")
            else:
                SeqIO.write([new_record], out_f, "fasta")
                seen_ids.add(new_record.id)


def main(
        alignments_dir: Path,
        output_dir: Path,
):
    """
    Search for all alignment files (*.aln.bz2) and transform them into FASTA files with all gaps removed.

    :param alignments_dir:
    :return:
    """
    output_dir.mkdir(exist_ok=True)
    all_markers_out_path = output_dir / "markers.fna"
    if all_markers_out_path.exists():
        print(f"Previous instance of {all_markers_out_path.name} found; it will be overwritten.")
        all_markers_out_path.unlink()

    seen_ids = set()
    for sgb_marker_file in alignments_dir.glob("*.fna.bz2"):
        gather_markers(sgb_marker_file, all_markers_out_path, seen_ids)


if __name__ == "__main__":
    main(
        Path("/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_database/PhyloPhlAn_output/markers_dna"),
        Path("/data/bwh-comppath-seq/youn/metaphlan_dset/phylophlan_database/processed/dna_only")
    )
