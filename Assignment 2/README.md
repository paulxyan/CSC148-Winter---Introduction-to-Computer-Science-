# CSC148 Assignment 2: Huffman Tree Compression

This project implements **lossless compression and decompression** using **Huffman coding**. It builds a Huffman tree from byte frequencies, generates prefix free codes, compresses file contents into a compact bitstream, and restores the original file by decoding with the stored tree.

---

## What this program does

- Reads an input file as bytes
- Computes symbol frequencies
- Builds a Huffman tree (including required edge cases such as single symbol inputs)
- Generates Huffman codes from the tree
- Writes a compressed file format that includes the encoded tree and compressed data
- Decompresses by reconstructing the tree and decoding the bitstream back to the original bytes

---

## Repository structure (suggested)

```text
csc148-a2-huffman-compression/
  README.md
  src/
    compress.py
    huffman.py
    utils.py
  tests/
    test_huffman_properties_basic.py
    test_huffman_properties_compression.py
  docs/
    assignment_handout.pdf
  plagiarism.txt
  archive/
    Copy3.py
    Refactored.py
    Refactored5.py
    compress - Copy.py
    PaulYansTest.py
  .gitignore
