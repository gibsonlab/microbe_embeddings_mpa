import sys
import subprocess
from typing import Set
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd


def parse_mothur_bad_ids(
        filepath: Path,
        remove_frac_below: float,
) -> Set[str]:
    align_report = pd.read_csv(filepath, sep="\t")
    query_len = align_report['QueryLength']
    query_start = align_report['QueryStart']
    query_end = align_report['QueryEnd']
    frac_aligned = (query_end - query_start + 1) / query_len
    bad_section = align_report.loc[frac_aligned < remove_frac_below]
    print("bad ASVs:", bad_section.shape[0])
    raise Exception("DEBUG!")


def convert_mother_alignment(mothur_aln_path: Path, out_path: Path, bad_ids: Set[str]):
    with open(mothur_aln_path, "rt") as read_f, out_path.open("wt") as out_f:
        for record in SeqIO.parse(read_f, "fasta"):
            if record.id in bad_ids:
                print(f"Skipping conversion of {record.id} to gapped FASTA output!")
                continue

            record.seq = Seq(str(record.seq).replace('.', '-'))
            SeqIO.write(record, out_f, "fasta")


def run_mothur(
        in_fasta: Path,
        out_fasta: Path,
        reference_16s_path: Path,
        n_processors: int = 20,
        remove_frac_below: float = 0.85,
        mothur_cmd: str = 'mothur'
) -> Set[str]:
    """
    Run the alignment.
    :return: the set of ASV ids that MOTHUR suggests removing  (due to too many trimmed bases)
    """
    asvs_bad_id_filepath = in_fasta.with_suffix('.flip.accnos')

    if out_fasta.exists():
        print(f"alignment output already exists: {out_fasta}")
        return parse_mothur_bad_ids(asvs_bad_id_filepath)

    if not in_fasta.exists():
        raise FileNotFoundError(f"Input file not found: {in_fasta}")

    # Check for MOTHUR installation
    try:
        subprocess.run(
            [mothur_cmd, "--version"],
            capture_output=True,
            check=True
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "MOTHUR is not installed or not in PATH. Install it with: conda install -c bioconda mothur"
        ) from None

    # Run MOTHUR
    # Example command: mothur "#align.seqs(fasta=asv_sequences.post_filter.fasta, reference=Ecoli_16s.fasta, processors=20)"
    try:
        exec_mothur_cmd = f"#align.seqs(fasta={str(in_fasta)}, reference={str(reference_16s_path)}, processors={n_processors})"
        print("Running mothur command: {}".format(exec_mothur_cmd))
        result = subprocess.run(
        [mothur_cmd, exec_mothur_cmd],
            cwd=in_fasta.parent,
            capture_output=True,
            text=True,
            check=True  # Raises exception if return code is non-zero
        )
        print("STDOUT:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running mothur: {e}")
        print(f"Return code: {e.returncode}")
        print(f"STDERR: {e.stderr}")
        raise

    align_report_filepath = in_fasta.with_suffix('.align_report')
    align_filepath = in_fasta.with_suffix('.align')
    bad_ids = parse_mothur_bad_ids(align_report_filepath, remove_frac_below=remove_frac_below)
    convert_mother_alignment(align_filepath, out_fasta, bad_ids)
    return bad_ids



def run_mafft(in_fasta: Path, out_fasta: Path, mafft_cmd: str = 'mafft'):
    """ Run the alignment. """
    if not out_fasta.exists():
        # Check if input file exists
        if not in_fasta.exists():
            raise FileNotFoundError(f"Input file not found: {in_fasta}")

        # Check if MAFFT is installed
        try:
            subprocess.run(
                [mafft_cmd, "--version"],
                capture_output=True,
                check=True
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "MAFFT is not installed or not in PATH. "
                "Install it with: conda install -c bioconda mafft"
            )

        # Run MAFFT
        # MAFFT writes to stdout by default, so we redirect it to the output file
        try:
            with open(out_fasta, 'w') as out_file:
                _ = subprocess.run(
                    [mafft_cmd, str(in_fasta)],
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )

            print(f"MAFFT alignment completed successfully: {out_fasta}")

        except subprocess.CalledProcessError as e:
            print(f"MAFFT failed with error:\n{e.stderr}", file=sys.stderr)
            raise
    else:
        print(f"alignment output already exists: {out_fasta}")
