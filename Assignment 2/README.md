# Huffman Tree Compression

This project implements lossless file compression and decompression using Huffman coding. It reads a file as bytes, builds a Huffman tree from symbol frequencies, encodes the content using a prefix free variable length code, and writes a compressed binary format that can be decoded back to the original file. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

## What is included

1) Core implementation in src
   * compress.py contains the main compression and decompression pipeline including compress_file and decompress_file :contentReference[oaicite:3]{index=3}
   * huffman.py provides the HuffmanTree class used across the implementation :contentReference[oaicite:4]{index=4}
   * utils.py provides helpers for working with bits and bytes and includes ReadNode which is used to reconstruct trees from compressed files :contentReference[oaicite:5]{index=5}

2) Tests in tests
   * Property based tests for correctness and round trip compression

3) Assignment reference in docs
   * The original handout describing required functions and the compressed file representation

## Key functions implemented

Suggested implementation order from the handout
* build_frequency_dict
* build_huffman_tree including the single symbol special case where a dummy leaf is added with frequency 0 :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
* get_codes
* number_nodes
* compress_bytes and the file level wrapper compress_file :contentReference[oaicite:8]{index=8}
* tree serialization and reconstruction helpers such as tree_to_bytes generate_tree_postorder and generate_tree_general :contentReference[oaicite:9]{index=9}
* decompress_bytes and the file level wrapper decompress_file :contentReference[oaicite:10]{index=10}
* improve_tree which optimizes symbol placement without changing tree shape :contentReference[oaicite:11]{index=11}

## How to run

Option A call the file wrappers directly from Python

```python
from compress import compress_file, decompress_file

compress_file("input.txt", "output.huf")
decompress_file("output.huf", "roundtrip.txt")
