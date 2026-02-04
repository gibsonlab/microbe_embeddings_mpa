"""
Script to identify 16S rRNA sequences from a FASTA file using BLAST.
Only keeps sequences where ALL top 5 hits are 16S sequences.
Requires: BioPython, BLAST+ command line tools
"""

from typing import *
from pathlib import Path

import subprocess
from Bio import SeqIO
from Bio.Blast import NCBIXML
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq


ASVSequenceDict = Dict[str, str]


def pipeline_16s_validation(
        asv_seqs: ASVSequenceDict,
        cache_dir: Path,
        blast_db: Path,
        blast_num_threads: int = 1,
        defer_to_hpc: bool = True,
) -> ASVSequenceDict:
    """
    Run BLAST validation to ensure ASVs are actually 16S sequences.
    Note: this may take anywhere between 6-24 hours (on Alienware aurora r16, 'nt' database hosted in /data/cctm), depending on query size and database disk i/o throughput.

    :return: A (Dict, Path) tuple.
    1) The dictionary of ASV IDs -> ASV Sequences mapping, restricted to those ASVs which pass the filter.
    2) The path to the resulting FASTA file containing only the 
    """
    asv_sequence_file_preblast = cache_dir / "asv_sequences.preblast.fasta"
    dict_to_fasta(asv_seqs, asv_sequence_file_preblast)

    validation_output = cache_dir / "asv_blast_validation.txt"
    rejected_output_path = cache_dir / "asv_blast_rejects.txt"

    validate_all_16s(
        input_fasta=asv_sequence_file_preblast,
        output_path=validation_output,
        rejected_output_path=rejected_output_path,
        blast_db=blast_db,
        n_threads=blast_num_threads,
        blastn_path='/data/local/youn/miniforge3/envs/evo/bin/blastn',
        defer_to_hpc=defer_to_hpc,
    )

    """
    Parse the file and refine the asv subset.
    """
    with open(validation_output, "rt") as f:
        asv_id_subset = set()
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            asv_id_subset.add(line)

    asv_seqs_subset = {asv_id: asv_seq for asv_id, asv_seq in asv_seqs.items() if asv_id in asv_id_subset}
    assert len(asv_seqs_subset) == len(asv_id_subset)
    return asv_seqs_subset


def dict_to_fasta(sequences_dict: ASVSequenceDict, output_file: Path):
    """
    Write sequences from dictionary to multi-FASTA file using BioPython.

    Parameters:
    -----------
    sequences_dict : dict
        Dictionary mapping sequence IDs to sequences
    output_file : str
        Path to output FASTA file
    """
    # Create SeqRecord objects
    seq_records = []
    for seq_id, sequence in sequences_dict.items():
        seq_record = SeqRecord(
            Seq(sequence),
            id=seq_id,
            description=""  # Empty description, just ID
        )
        seq_records.append(seq_record)

    # Write to FASTA file
    with open(output_file, "w") as handle:
        SeqIO.write(seq_records, handle, "fasta")

    print(f"Wrote {len(seq_records)} sequences to {output_file}")


def check_blast_installation(blastn_path='blastn'):
    """Check if BLAST+ is installed and accessible."""
    try:
        result = subprocess.run([blastn_path, '-version'],
                                capture_output=True, text=True, check=True)
        print(f"BLAST+ found: {result.stdout.split()[1]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: BLAST+ not found. Please install NCBI BLAST+ tools.")
        print("Download from: https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download")
        return False


def blast_sequences(
        input_fasta: Path,
        output_xml: Path,
        blast_db: Union[str, Path],
        blastn_path: str = 'blastn',
        n_threads: int = 1,
        use_remote: bool = True,
        min_identity: int = 95,
        print_hpc_instructions: bool = True,
):
    """Run BLAST search using subprocess."""
    print(f"Running BLAST search on {input_fasta} against {blast_db}...")

    # Build BLAST command
    blast_cmd = [
        blastn_path,
        '-task', 'megablast',
        '-query', str(input_fasta),
        '-db', str(blast_db),
        '-outfmt', '5',  # XML output
        '-out', str(output_xml),
        '-evalue', '1e-10',
        '-max_target_seqs', '5',
        '-perc_identity', str(min_identity),
        '-max_hsps', '1',
        '-mt_mode', '0',
        '-num_threads', str(n_threads),
    ]

    # Add remote flag if using NCBI remote BLAST
    if use_remote:
        blast_cmd.append('-remote')

    if output_xml.exists() and output_xml.stat().st_size > 0:
        print(f"BLAST raw output {output_xml} already exists!")
        return
    else:
        if print_hpc_instructions:
            print("[NOTICE] Please run the following command on an HPC compute node: 'blastn {}'".format(
                ' '.join(blast_cmd[1:])
            ))
            raise Exception("BLAST was not run. Please read the above printed message!")

        try:
            print(f"Running command: {' '.join(blast_cmd)}")
            result = subprocess.run(
                blast_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stderr:
                print(f"BLAST stderr: {result.stderr}")

            print("BLAST search completed.")
            return True

        except subprocess.TimeoutExpired:
            raise RuntimeError("Error: BLAST search timed out")
        except subprocess.CalledProcessError as e:
            print(f"BLAST stderr: {e.stderr}")
            raise RuntimeError(f"Error running BLAST: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error running BLAST: {e}")


def is_16s_sequence(hit_description):
    """Check if a BLAST hit description indicates a 16S sequence."""
    hit_desc = hit_description.lower()

    # Keywords that indicate 16S rRNA
    positive_keywords = [
        '16s ribosomal rna', '16s rrna', '16s ribosomal',
        '16s small subunit', 'small subunit ribosomal rna',
        'ssu rrna', '16s rdna', 'small subunit rrna'
    ]

    # Keywords that indicate NOT 16S (18S, 23S, 28S, etc.)
    negative_keywords = [
        '18s', '23s', '28s', '5s', '5.8s',
        'large subunit', 'lsu', 'its1', 'its2',
        'internal transcribed spacer', 'chloroplast',
        'mitochondrial', 'plastid'
    ]

    # First check for negative keywords - if found, it's not 16S
    for neg_keyword in negative_keywords:
        if neg_keyword in hit_desc:
            return False

    # Then check for positive keywords
    for pos_keyword in positive_keywords:
        if pos_keyword in hit_desc:
            return True

    return False


def parse_blast_results(xml_file: Path, min_identity: int = 80):
    """
    Parse BLAST XML results and identify sequences where ALL top 5 hits are 16S.
    """
    print("Parsing BLAST results...")

    confirmed_16s = set()
    rejected_sequences = {}

    try:
        with open(xml_file, 'r') as result_handle:
            blast_records = NCBIXML.parse(result_handle)

            for blast_record in blast_records:
                query_id = blast_record.query.split()[0]
                query_length = blast_record.query_length

                # Get top 5 alignments
                top_alignments = blast_record.alignments[:5]

                if len(top_alignments) == 0:
                    rejected_sequences[query_id] = "No BLAST hits found"
                    continue

                # Check each of the top 5 hits
                hit_details = []
                all_hits_are_16s = True

                for i, alignment in enumerate(top_alignments):
                    hit_def = alignment.hit_def

                    # Get the best HSP for this alignment
                    if alignment.hsps:
                        hsp = alignment.hsps[0]  # Best HSP
                        identity = (hsp.identities / hsp.align_length) * 100

                        # Skip hits with very low identity
                        if identity < min_identity:
                            continue

                        is_16s = is_16s_sequence(hit_def)
                        hit_details.append({
                            'rank': i + 1,
                            'identity': identity,
                            'description': hit_def[:100] + "..." if len(hit_def) > 100 else hit_def,
                            'is_16s': is_16s
                        })

                        if not is_16s:
                            all_hits_are_16s = False

                # Decision: keep only if ALL top hits are 16S
                if len(hit_details) == 0:
                    rejected_sequences[query_id] = f"No hits above identity threshold {min_identity}"
                elif all_hits_are_16s and len(hit_details) > 0:
                    confirmed_16s.add(query_id)
                else:
                    # Find the first non-16S hit for reporting
                    non_16s_hits = [hit for hit in hit_details if not hit['is_16s']]
                    if non_16s_hits:
                        first_non_16s = non_16s_hits[0]
                        rejected_sequences[
                            query_id] = f"Hit {first_non_16s['rank']} is not 16S: {first_non_16s['description']}"

    except Exception as e:
        print(f"Error parsing BLAST results: {e}")
        return set(), {}

    return confirmed_16s, rejected_sequences


def validate_all_16s(
        input_fasta: Path,
        output_path: Path,
        rejected_output_path: Path,
        blast_db: Path,
        n_threads: int = 1,
        blastn_path: str = 'blastn',
        min_pct_identity: int = 95,
        database: str = 'nt',
        defer_to_hpc: bool = True,
):
    # Check if input file exists
    if not input_fasta.exists():
        raise FileNotFoundError(f"Error: Input file '{input_fasta}' not found.")

    # Check BLAST installation
    if not check_blast_installation(blastn_path=blastn_path):
        raise Exception("BLAST is not installed.")

    # Count input sequences
    seq_count = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    print(f"Found {seq_count} sequences in input file.")
    print(f"Using database: {database}")
    print(f"Minimum identity threshold: {min_pct_identity}%")
    print("Strategy: Only keeping sequences where ALL top 5 hits are 16S\n")

    # Modify BLAST command for local vs remote
    blast_output = output_path.parent / "blast_results.xml"

    # Run BLAST
    blast_sequences(
        input_fasta,
        blast_output,
        blast_db=blast_db,
        blastn_path=blastn_path,
        n_threads=n_threads,
        use_remote=False,
        min_identity=min_pct_identity,
        print_hpc_instructions=defer_to_hpc,
    )

    # Parse results
    confirmed_16s_ids, rejected_sequences = parse_blast_results(
        blast_output,
        min_identity=min_pct_identity
    )

    # Write confirmed 16S results
    with open(output_path, 'w') as f:
        for seq_id in sorted(confirmed_16s_ids):
            f.write(f"{seq_id}\n")

    # Write rejected sequences with reasons
    with open(rejected_output_path, 'w') as f:
        f.write("Sequence_ID\tReason_for_Rejection\n")
        for seq_id in sorted(rejected_sequences.keys()):
            f.write(f"{seq_id}\t{rejected_sequences[seq_id]}\n")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total sequences analyzed: {seq_count}")
    print(f"Confirmed 16S sequences: {len(confirmed_16s_ids)}")
    print(f"Rejected sequences: {len(rejected_sequences)}")
    print(f"Success rate: {len(confirmed_16s_ids) / seq_count * 100:.1f}%")
    print(f"\nConfirmed 16S IDs saved to: {output_path}")
    print(f"Rejected sequences saved to: {rejected_output_path}")
