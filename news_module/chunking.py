"""
Intelligent chunking to keep LLM payloads under context limits (avoids 413s)
while retaining semantically dense content (Phase III requirement).

Approach: rough token estimate (chars/4, a standard approximation for
English text across GPT/Gemini/Llama tokenizers -- good enough to stay
safely under budget without pulling in a model-specific tokenizer per tier).
Split on paragraph boundaries first, falling back to sentence boundaries,
so we never cut a sentence mid-word, which would lose extractable entities
right at a chunk edge.
"""
import re


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int, overlap_tokens: int = 200) -> list[str]:
    if estimate_tokens(text) <= max_tokens:
        return [text]

    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                # single paragraph too big -> split on sentences
                sentences = re.split(r"(?<=[.!?])\s+", para)
                sub = ""
                for s in sentences:
                    if len(sub) + len(s) + 1 <= max_chars:
                        sub = f"{sub} {s}" if sub else s
                    else:
                        if sub:
                            chunks.append(sub)
                        sub = s
                current = sub
            else:
                current = para

    if current:
        chunks.append(current)

    # add overlap between consecutive chunks so entities split across a
    # boundary still appear whole in at least one chunk
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap_chars:]
            overlapped.append(tail + "\n\n" + chunks[i])
        chunks = overlapped

    return chunks
