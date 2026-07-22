# Security

Threat boundaries include untrusted configuration, prompts, dataset records, filenames, provider output and local artifacts. Path resolution rejects absolute paths by default, traversal, symlink escapes and Windows reserved names. Resource limits bound messages, characters, output tokens, files, records, reports and stream events.

Secret detection covers common API-key, bearer, password, credential and private-key forms while avoiding broad rejection of ordinary technical prose. Logging redacts sensitive assignments. These controls reduce accidental exposure but do not replace OS permissions, sandboxing, malware scanning or a security review.

