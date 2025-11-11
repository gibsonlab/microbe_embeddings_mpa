"""
Step 1: Take the multiple alignment file output by PhyloPhlAn (amino acid alignments) and remove all gaps from each
sequence entry. This recovers the amino acid sequences of the gene markers.
"""
from pathlib import Path
import bz2
import zstandard as zstd

from Bio import SeqIO
from Bio.bgzf import BgzfWriter
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def remove_gaps_from_seq(x: Seq) -> Seq:
    return Seq(str(x).replace("-", ""))


def remove_gaps_from_file(alignment_file_bz2: Path, out_bgzf: Path, gene_name: str):
    print("De-gapping src: {}, dest: {}".format(
        alignment_file_bz2,
        out_bgzf,
    ))

    with (
        bz2.open(alignment_file_bz2, "rt") as aln_f,
        BgzfWriter(out_bgzf, "ab") as out_f
    ):
        for record in SeqIO.parse(aln_f, "fasta"):
            ungapped_seq = remove_gaps_from_seq(record.seq)
            new_record = SeqRecord(ungapped_seq, id=record.id, description=gene_name)
            SeqIO.write([new_record], out_f, "fasta")


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
    all_bgzf = output_dir / "all_markers.fna.bgz"
    if all_bgzf.exists():
        all_bgzf.unlink()  # delete the file, possibly from a previous run.

    for alignment_file_bz2 in alignments_dir.glob("*.aln.bz2"):
        gene_name = alignment_file_bz2.name[:-len(".aln.bz2")]
        remove_gaps_from_file(alignment_file_bz2, all_bgzf, gene_name)


if __name__ == "__main__":
    main(
        Path("/data/cctm/youn/metaphlan_dset/phylophlan_data/PhyloPhlAn_output/msas"),
        Path("/data/cctm/youn/metaphlan_dset/phylophlan_data/processed")
    )
