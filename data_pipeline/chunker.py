"""
Chunking & text normalization module for the Ashen Era Archive.
Splits large text documents and long pages into optimal RAG chunks
while preserving document metadata.
"""

from typing import List, Dict, Any


def split_text_recursive(
    text: str,
    max_chunk_size: int = 2000,
    chunk_overlap: int = 200,
    separators: List[str] = None,
) -> List[str]:
    """
    Recursively splits text into chunks under max_chunk_size with overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chunk_size:
        return [text]

    # Find the first applicable separator
    chosen_sep = ""
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            break

    splits = text.split(chosen_sep) if chosen_sep else list(text)
    chunks: List[str] = []
    current_chunk = ""

    for piece in splits:
        piece_to_add = (chosen_sep + piece) if current_chunk and chosen_sep else piece
        if len(current_chunk) + len(piece_to_add) <= max_chunk_size:
            current_chunk += piece_to_add
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If a single piece is larger than max_chunk_size, recursively split it with finer separators
            if len(piece) > max_chunk_size:
                remaining_seps = separators[separators.index(chosen_sep) + 1 :] if chosen_sep in separators else [""]
                sub_splits = split_text_recursive(piece, max_chunk_size, chunk_overlap, remaining_seps)
                chunks.extend(sub_splits)
                current_chunk = ""
            else:
                # Retain overlap from end of previous chunk if possible
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    overlap_text = current_chunk[-chunk_overlap:]
                    current_chunk = overlap_text + piece_to_add
                else:
                    current_chunk = piece

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if c.strip()]


def normalize_and_chunk_dataset(
    raw_chunks: List[Dict[str, Any]],
    max_chunk_size: int = 2000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Normalizes raw extracted JSON chunks into uniformly sized chunks.
    """
    processed_chunks: List[Dict[str, Any]] = []
    chunk_counter = 0

    for item in raw_chunks:
        source = item.get("source", "unknown")
        page = item.get("page", 1)
        doc_type = item.get("type", "document")
        content = item.get("content", "").strip()

        if not content:
            continue

        # If content fits in max_chunk_size, keep as single chunk
        if len(content) <= max_chunk_size:
            chunk_counter += 1
            processed_chunks.append({
                "chunk_id": f"chunk_{chunk_counter:05d}",
                "source": source,
                "page": page,
                "type": doc_type,
                "sub_index": 0,
                "char_count": len(content),
                "content": content,
            })
        else:
            sub_pieces = split_text_recursive(
                content,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for sub_idx, piece in enumerate(sub_pieces):
                chunk_counter += 1
                processed_chunks.append({
                    "chunk_id": f"chunk_{chunk_counter:05d}",
                    "source": source,
                    "page": page,
                    "type": doc_type,
                    "sub_index": sub_idx,
                    "char_count": len(piece),
                    "content": piece,
                })

    return processed_chunks
