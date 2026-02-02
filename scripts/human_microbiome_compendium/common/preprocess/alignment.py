import sys
import subprocess
from pathlib import Path


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
