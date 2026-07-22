# Datasets

A manifest identifies dataset/version/split, count, language, license, SHA-256 checksum and source. Records require stable IDs, output, source and language; optional instruction/input/messages and limited metadata support common supervised formats.

Validation rejects empty output, duplicates, invalid languages, excessive size, private indicators and common secret patterns. Text preprocessing uses NFC normalization and whitespace cleanup; fingerprints support duplicate detection. Data owners must establish license, consent, provenance, personal-data removal and retention policy. Repository ignore rules exclude actual data.

