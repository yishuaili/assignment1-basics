import regex as re

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import Counter
from collections.abc import Iterable
import concurrent.futures

# from cs336_basics.utils.io import GPT2_PRETOKENIZER_PATTERN
from cs336_basics.pretokenization_example import find_chunk_boundaries

GPT2_PRETOKENIZER_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _find_pretokens(text: str):
    """
    Find the pretokens in the text.
    """
    logging.info(f"Pre-tokenizing the text of length {len(text)}")
    return Counter(re.findall(GPT2_PRETOKENIZER_PATTERN, text))


def _read_text_file(input_path: str, num_worker: int, special_tokens: Iterable[str]):
    """
    Read the text file at the given path.
    Return the text as pretoken frequency table.
    """
    # Read the input text file
    with open(input_path, "r") as file:
        text = file.read()
    
    # Remove special tokens from the text
    for token in special_tokens:
        text = text.replace(token, "")

    logging.info("Initializing pretoken frequency table")
    if num_worker == 1:
        pretokens = _find_pretokens(text)
    else:
        boundaries = find_chunk_boundaries(text, num_worker, "<|endoftext|>".encode("utf-8"))
        text_chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            text.seek(start)
            chunk = text.read(end - start).decode("utf-8", errors="ignore")
            text_chunks.append(chunk)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_worker) as executor:
            pretokens = Counter(executor.map(_find_pretokens, text_chunks))
        pretokens = sum(pretokens, Counter())
    
    gen_tuple_of_bytes = lambda pretoken: tuple([bytes([b]) for b in pretoken.encode("utf-8")])
    pretoken_freq = {}
    for pretoken, freq in pretokens.items():
        pretoken_freq[gen_tuple_of_bytes(pretoken)] = freq

    return pretoken_freq


def _update_byte_tuple(byte_tuple: Iterable[bytes], merge_loc: int):
    """
    Merge the byte tuple at the merge location.
    """
    assert len(byte_tuple) > 1, "Cannot merge a byte tuple with length less than 2."
    prefix = byte_tuple[:merge_loc]
    tomerge = byte_tuple[merge_loc:merge_loc+2]
    suffix = byte_tuple[merge_loc+2:]
    new_byte_tuple = prefix + (b"".join(tomerge),) + suffix
    return new_byte_tuple, prefix, suffix



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_workers: int = 1, **kwargs) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    """
    # Initialize the vocab with 256 bytes and sepcial tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[256+i] = token.encode("utf-8")

    pretoken_freq = _read_text_file(input_path, num_workers, special_tokens)

    logging.info("Initializing byte pair frequency table")
    pair_freq = Counter()
    for pretoken_tuple, freq in tqdm(pretoken_freq.items(), disable=False):
        for i in range(len(pretoken_tuple) - 1):
            pair = pretoken_tuple[i:i+2]
            if pair not in pair_freq:
                pair_freq[pair] = 0
            pair_freq[pair] += freq

    logging.info("Performing BPE algorithm")
    pre_merge_vocab_size = len(vocab)
    pbar = tqdm(total=vocab_size-pre_merge_vocab_size)

    merges = []
    while len(vocab) < vocab_size:
        # Find the most frequent pair
        most_freq_pair = max(pair_freq, key=lambda k: (pair_freq[k], k))

        # Add the pair to the merges list
        merges.append(most_freq_pair)
        
        # Update the vocab
        new_id = max(vocab.keys()) + 1
        vocab[new_id] = b"".join(most_freq_pair)

        # Update the pre-token frequency table and pair frequency table
        new_pretoken_freq = {}
        for pretoken_tuple, freq in pretoken_freq.items():
            i=0
            while i < len(pretoken_tuple):
                pair = pretoken_tuple[i:i+2]
                if pair == most_freq_pair:
                    pretoken_tuple, prefix, suffix = _update_byte_tuple(pretoken_tuple, i)

                    # Update the pair frequency table
                    if prefix:
                        add_pair = (prefix[-1], vocab[new_id])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (prefix[-1], most_freq_pair[0])
                        pair_freq[del_pair] -= freq
                    if suffix:
                        add_pair = (vocab[new_id], suffix[0])
                        pair_freq[add_pair] = pair_freq.get(add_pair, 0) + freq
                        del_pair = (most_freq_pair[1], suffix[0])
                        pair_freq[del_pair] -= freq
                    pair_freq[most_freq_pair] -= freq
                i+=1
            # Update the pre-token frequency table
            new_pretoken_freq[pretoken_tuple] = freq
        pretoken_freq = new_pretoken_freq
        pbar.update(len(vocab) - pre_merge_vocab_size - pbar.n) 
    pbar.close()

    return vocab, merges

