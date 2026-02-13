"""
Script to identify bacterial 16S rRNA sequences from a FASTA file using VSEARCH.
Only keeps sequences that match bacterial 16S sequences in a reference database.
Filters out Archaea, Eukaryota, and other non-bacterial domains.
Requires: BioPython, vsearch
"""

import subprocess
from typing import Union
from pathlib import Path

from Bio import SeqIO
from .asvs import dict_to_fasta


def pipeline_16s_validation(
        asv_seqs: dict[str, str],
        cache_dir: Path,
        silva_db: Path,
        vsearch_path: Union[str, Path] = "vsearch",
        vsearch_num_threads: int = 1,
        min_identity: float = 0.90,
) -> dict[str, str]:
    """
    Run VSEARCH validation to ensure ASVs are bacterial 16S sequences.
    Filters out Archaea, Eukaryota, and other non-bacterial domains.
    Note: this is MUCH faster than BLAST (minutes instead of hours).

    :param asv_seqs: Dictionary mapping ASV IDs to sequences
    :param cache_dir: Directory for caching intermediate files
    :param silva_db: Path to SILVA or Greengenes 16S database (FASTA format)
    :param vsearch_path: Path to vsearch executable
    :param vsearch_num_threads: Number of threads for VSEARCH
    :param min_identity: Minimum identity threshold (0.0-1.0)
    :return: Dictionary of ASV IDs -> ASV Sequences mapping, restricted to bacterial ASVs.
    """
    asv_sequence_file = cache_dir / "asv_sequences.pre_validation.fasta"
    dict_to_fasta(asv_seqs, asv_sequence_file)

    validation_output = cache_dir / "asv_vsearch_validation.txt"
    rejected_output_path = cache_dir / "asv_vsearch_rejects.txt"

    # Check VSEARCH installation
    if not check_vsearch_installation(vsearch_path=str(vsearch_path)):
        raise Exception("VSEARCH is not installed.")

    validate_all_16s_vsearch(
        input_fasta=asv_sequence_file,
        output_path=validation_output,
        rejected_output_path=rejected_output_path,
        silva_db=silva_db,
        n_threads=vsearch_num_threads,
        min_identity=min_identity,
        vsearch_path=str(vsearch_path),
    )

    # Parse the file and refine the asv subset
    with open(validation_output, "rt") as f:
        asv_id_subset: set[str] = set()
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            asv_id_subset.add(line)

    asv_seqs_subset: dict[str, str] = {
        asv_id: asv_seq 
        for asv_id, asv_seq in asv_seqs.items() 
        if asv_id in asv_id_subset
    }
    assert len(asv_seqs_subset) == len(asv_id_subset)
    return asv_seqs_subset


def check_vsearch_installation(vsearch_path: str = 'vsearch') -> bool:
    """
    Check if VSEARCH is installed and accessible.
    
    :param vsearch_path: Path to vsearch executable
    :return: True if vsearch is found, False otherwise
    """
    try:
        result: subprocess.CompletedProcess = subprocess.run(
            [vsearch_path, '--version'],
            capture_output=True, 
            text=True, 
            check=True
        )
        # VSEARCH prints version to stderr
        version_info = result.stderr.split('\n')[0] if result.stderr else result.stdout.split('\n')[0]
        print(f"VSEARCH found: {version_info}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: VSEARCH not found. Please install VSEARCH.")
        print("Install via conda: conda install -c bioconda vsearch")
        return False


def run_vsearch(
        input_fasta: Path,
        output_tsv: Path,
        silva_db: Path,
        vsearch_path: str = 'vsearch',
        n_threads: int = 1,
        min_identity: float = 0.85,
        max_accepts: int = 5,
) -> bool | None:
    """
    Run VSEARCH usearch_global to search sequences against reference database.
    
    :param input_fasta: Input FASTA file with query sequences
    :param output_tsv: Output TSV file (BLAST6 format)
    :param silva_db: Reference database (SILVA or Greengenes FASTA)
    :param vsearch_path: Path to vsearch executable
    :param n_threads: Number of threads
    :param min_identity: Minimum identity (0.0-1.0)
    :param max_accepts: Maximum number of hits to accept per query
    :return: True if successful, None if output already exists
    """
    print(f"Running VSEARCH on {input_fasta} against {silva_db}...")

    # Build VSEARCH command
    vsearch_cmd: list[str] = [
        vsearch_path,
        '--usearch_global', str(input_fasta),
        '--db', str(silva_db),
        '--id', str(min_identity),
        '--maxaccepts', str(max_accepts),
        '--maxrejects', '32',
        '--threads', str(n_threads),
        '--blast6out', str(output_tsv),
        '--top_hits_only',
    ]

    if output_tsv.exists() and output_tsv.stat().st_size > 0:
        print(f"VSEARCH output {output_tsv} already exists!")
        return None
    
    try:
        print(f"Running command: {' '.join(vsearch_cmd)}")
        result: subprocess.CompletedProcess = subprocess.run(
            vsearch_cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stderr:
            print(f"VSEARCH stderr:\n{result.stderr}")
        if result.stdout:
            print(f"VSEARCH stdout:\n{result.stdout}")

        print("VSEARCH search completed successfully.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n{'='*60}")
        print(f"ERROR: VSEARCH failed with exit code {e.returncode}")
        print(f"{'='*60}")
        print(f"Command: {' '.join(vsearch_cmd)}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        print(f"{'='*60}\n")
        raise RuntimeError(f"VSEARCH failed with exit code {e.returncode}: {e.stderr}")
    except FileNotFoundError as e:
        print(f"\nERROR: VSEARCH executable not found at '{vsearch_path}'")
        print("Make sure VSEARCH is installed: conda install -c bioconda vsearch")
        raise RuntimeError(f"VSEARCH executable not found: {e}")
    except Exception as e:
        print(f"\nERROR: Unexpected error running VSEARCH")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        raise RuntimeError(f"Unexpected error running VSEARCH: {e}")


def extract_domain_from_taxonomy(target_id: str) -> str:
    """
    Extract domain (Bacteria, Archaea, Eukaryota) from SILVA or Greengenes taxonomy string.
    
    In VSEARCH BLAST6 output, the target field (column 1) contains the taxonomy string directly.
    
    Examples:
    - "Bacteria;Proteobacteria;Alphaproteobacteria;..."
    - "Archaea;Euryarchaeota;..."
    - "k__Bacteria;p__Proteobacteria;c__Gammaproteobacteria;..." (Greengenes)
    
    :param target_id: Taxonomy string from VSEARCH output (column 1)
    :return: Domain name (Bacteria, Archaea, Eukaryota, or Unknown)
    """
    # The target_id is the taxonomy string itself
    taxonomy_string = target_id.strip()
    
    if not taxonomy_string:
        return "Unknown"
    
    # Split taxonomy by semicolon to get levels
    taxonomy_levels = taxonomy_string.split(';')
    
    if len(taxonomy_levels) == 0:
        return "Unknown"
    
    # Get the first taxonomy level (domain)
    first_level = taxonomy_levels[0].strip()
    
    if not first_level:
        return "Unknown"
    
    # Handle Greengenes format: "k__Bacteria" -> "Bacteria"
    if first_level.startswith('k__'):
        domain = first_level[3:]  # Remove "k__" prefix
    # Handle SILVA/direct format: "Bacteria" (already clean)
    else:
        domain = first_level
    
    # Normalize and classify
    domain_lower = domain.lower()
    
    if domain_lower == 'bacteria':
        return "Bacteria"
    elif domain_lower in ['archaea', 'archaebacteria']:
        return "Archaea"
    elif domain_lower in ['eukaryota', 'eukarya']:
        return "Eukaryota"
    else:
        # Return the actual value found for debugging
        return f"Unknown ({domain})"


def parse_vsearch_results(
        tsv_file: Path,
        min_identity: float = 0.85,
) -> tuple[set[str], dict[str, tuple[str, str]]]:
    """
    Parse VSEARCH BLAST6 output and identify sequences with bacterial 16S hits.
    Filters out Archaea, Eukaryota, and other non-bacterial domains.
    
    BLAST6 format columns:
    0: query, 1: target, 2: identity, 3: alignment_length, 4: mismatches, 5: gaps,
    6: qstart, 7: qend, 8: tstart, 9: tend, 10: evalue, 11: bitscore
    
    :param tsv_file: VSEARCH output file in BLAST6 format
    :param min_identity: Minimum identity threshold (0.0-1.0, as percentage in file)
    :return: Tuple of (confirmed_bacterial_16s, sequence_domain_info)
             sequence_domain_info maps query_id -> (domain, reason)
    """
    print("Parsing VSEARCH results and filtering for bacterial sequences...")

    confirmed_bacterial: set[str] = set()
    sequence_hits: dict[str, list[dict]] = {}
    sequence_domains: dict[str, set[str]] = {}  # Track all domains hit by each query
    
    try:
        with open(tsv_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                fields: list[str] = line.strip().split('\t')
                if len(fields) < 12:
                    continue
                
                query_id: str = fields[0]
                target_id: str = fields[1]
                identity: float = float(fields[2]) / 100.0  # Convert percentage to fraction
                align_length: int = int(fields[3])
                evalue: float = float(fields[10])
                bitscore: float = float(fields[11])
                
                # Skip hits below identity threshold
                if identity < min_identity:
                    continue
                
                # Extract domain from target taxonomy
                domain: str = extract_domain_from_taxonomy(target_id)
                
                # Track domains for this query
                if query_id not in sequence_domains:
                    sequence_domains[query_id] = set()
                sequence_domains[query_id].add(domain)
                
                # Store hit information
                if query_id not in sequence_hits:
                    sequence_hits[query_id] = []
                
                sequence_hits[query_id].append({
                    'target': target_id,
                    'domain': domain,
                    'identity': identity,
                    'align_length': align_length,
                    'evalue': evalue,
                    'bitscore': bitscore,
                })
    
    except Exception as e:
        print(f"Error parsing VSEARCH results: {e}")
        return set(), {}
    
    # Build classification for each sequence
    sequence_domain_info: dict[str, tuple[str, str]] = {}
    
    for query_id, domains in sequence_domains.items():
        if "Bacteria" in domains:
            # Accept if any bacterial hit exists
            confirmed_bacterial.add(query_id)
            if len(domains) > 1:
                other_domains = domains - {"Bacteria"}
                sequence_domain_info[query_id] = ("Bacteria", f"Mixed hits: Bacteria + {', '.join(other_domains)}")
            else:
                sequence_domain_info[query_id] = ("Bacteria", "Bacterial hits only")
        elif "Archaea" in domains:
            sequence_domain_info[query_id] = ("Archaea", "Rejected: Archaeal sequence")
        elif "Eukaryota" in domains:
            sequence_domain_info[query_id] = ("Eukaryota", "Rejected: Eukaryotic sequence")
        elif "Unknown" in domains:
            sequence_domain_info[query_id] = ("Unknown", "Rejected: Domain could not be determined")
        else:
            sequence_domain_info[query_id] = ("Unknown", "Rejected: No recognizable domain")
    
    return confirmed_bacterial, sequence_domain_info


def validate_all_16s_vsearch(
        input_fasta: Path,
        output_path: Path,
        rejected_output_path: Path,
        silva_db: Path,
        min_identity: float,
        max_accepts: int = 5,
        n_threads: int = 1,
        vsearch_path: str = 'vsearch',
) -> None:
    """
    Validate sequences as bacterial 16S using VSEARCH against SILVA/Greengenes database.
    Filters out Archaea, Eukaryota, and other non-bacterial domains.
    
    :param input_fasta: Input FASTA file with sequences to validate
    :param output_path: Output file for confirmed bacterial 16S sequence IDs
    :param rejected_output_path: Output file for rejected sequences with reasons
    :param silva_db: Path to SILVA or Greengenes 16S database (FASTA format)
    :param min_identity: Minimum identity threshold (0.0-1.0)
    :param max_accepts: Maximum number of hits to accept per query
    :param n_threads: Number of threads for VSEARCH
    :param vsearch_path: Path to vsearch executable
    """
    # Check if input file exists
    if not input_fasta.exists():
        raise FileNotFoundError(f"Error: Input file '{input_fasta}' not found.")
    
    # Check if database exists
    if not silva_db.exists():
        raise FileNotFoundError(f"Error: Database file '{silva_db}' not found.")

    # Count input sequences
    seq_count: int = sum(1 for _ in SeqIO.parse(input_fasta, "fasta"))
    all_seq_ids: set[str] = {record.id for record in SeqIO.parse(input_fasta, "fasta")}
    
    print(f"Found {seq_count} sequences in input file.")
    print(f"Using database: {silva_db}")
    print(f"Minimum identity threshold: {min_identity * 100:.1f}%")
    print("Strategy: Keeping only bacterial sequences, filtering out Archaea and Eukaryota\n")

    # Run VSEARCH
    vsearch_output: Path = output_path.parent / "vsearch_results.tsv"
    run_vsearch(
        input_fasta,
        vsearch_output,
        silva_db=silva_db,
        vsearch_path=vsearch_path,
        n_threads=n_threads,
        min_identity=min_identity,
        max_accepts=max_accepts,
    )

    # Parse results with domain filtering
    confirmed_bacterial_ids, sequence_domain_info = parse_vsearch_results(
        vsearch_output,
        min_identity=min_identity,
    )

    # Build comprehensive rejection reasons
    rejected_sequences: dict[str, str] = {}
    domain_counts: dict[str, int] = {"Bacteria": 0, "Archaea": 0, "Eukaryota": 0, "Unknown": 0, "No hits": 0}
    
    for seq_id in all_seq_ids:
        if seq_id in confirmed_bacterial_ids:
            domain_counts["Bacteria"] += 1
        elif seq_id in sequence_domain_info:
            domain, reason = sequence_domain_info[seq_id]
            rejected_sequences[seq_id] = reason
            domain_counts[domain] += 1
        else:
            # No hits at all
            rejected_sequences[seq_id] = f"No VSEARCH hit above {min_identity * 100:.1f}% identity threshold"
            domain_counts["No hits"] += 1

    # Write confirmed bacterial results
    with open(output_path, 'w') as f:
        for seq_id in sorted(confirmed_bacterial_ids):
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
    print(f"Confirmed bacterial sequences: {domain_counts['Bacteria']}")
    print(f"Rejected - Archaea: {domain_counts['Archaea']}")
    print(f"Rejected - Eukaryota: {domain_counts['Eukaryota']}")
    print(f"Rejected - Unknown domain: {domain_counts['Unknown']}")
    print(f"Rejected - No hits: {domain_counts['No hits']}")
    print(f"Total rejected: {len(rejected_sequences)}")
    print(f"Success rate: {domain_counts['Bacteria'] / seq_count * 100:.1f}%")
    print(f"\nConfirmed bacterial IDs saved to: {output_path}")
    print(f"Rejected sequences saved to: {rejected_output_path}")