from typing import Iterator, Tuple
from pathlib import Path
import zstandard as zstd


def get_asv_sequences(asv_sequence_file: Path) -> Iterator[Tuple[str, str]]:
    """
    Fetch all of the ASV sequences from the asv_sequence_file location.
    :return: A generator over (<ASV_ID>, <ASV_SEQ>) tuples.
    """
    with zstd.open(asv_sequence_file, "rt") as f:
        header_line = f.readline()
        assert header_line.startswith("#OTU"), "Expected file to start with the header: `#OTU`"

        for line in f:
            tokens = line.strip().split("\t")
            assert len(tokens) == 2, f"Expected exactly two tokens. Got: {line}"

            asv_id, asv_seq = tokens[0], tokens[1]
            assert asv_id.startswith("ASV"), f"Expected token #1 to have prefix `ASV`. Got: {asv_id}"
            yield asv_id, asv_seq