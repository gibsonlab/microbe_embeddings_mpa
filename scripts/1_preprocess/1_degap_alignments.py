"""
Step 1: Take the multiple alignment file output by PhyloPhlAn (amino acid alignments) and remove all gaps from each
sequence entry. This recovers the amino acid sequences of the gene markers.
"""
from pathlib import Path
import bz2
from typing import *

from Bio import SeqIO
from Bio.bgzf import BgzfWriter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def remove_gaps_from_seq(x: Seq) -> Seq:
    return Seq(str(x).replace("-", ""))


def remove_gaps_from_file(alignment_file_bz2: Path, out_bgzf_file: BgzfWriter, gene_name: str, seen_ids: Set[str]):
    print(f"De-gapping src: {alignment_file_bz2}")

    with bz2.open(alignment_file_bz2, "rt") as aln_f:
        for record in SeqIO.parse(aln_f, "fasta"):
            ungapped_seq = remove_gaps_from_seq(record.seq)
            new_record = SeqRecord(ungapped_seq, id=f"{gene_name}:{record.id}", description="")
            if new_record.id in seen_ids:
                print(f"[WARNING] {new_record.id} already in use. Skipping!")
            else:
                SeqIO.write([new_record], out_bgzf_file, "fasta")
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
    all_bgzf_path = output_dir / "all_markers.fna.bgz"
    if all_bgzf_path.exists():
        print(f"Previous instance of {all_bgzf_path.name} found; it will be overwritten.")

    seen_ids = set()
    with BgzfWriter(all_bgzf_path, "wb") as all_bgzf_out:
        for alignment_file_bz2 in alignments_dir.glob("*.aln.bz2"):
            gene_name = alignment_file_bz2.name[:-len(".aln.bz2")]
            remove_gaps_from_file(alignment_file_bz2, all_bgzf_out, gene_name, seen_ids)


if __name__ == "__main__":
    main(
        Path("/data/cctm/youn/metaphlan_dset/phylophlan_data/PhyloPhlAn_output/msas"),
        Path("/data/cctm/youn/metaphlan_dset/phylophlan_data/processed")
    )
