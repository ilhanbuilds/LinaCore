# Security

Güven sınırları untrusted config, prompt, manifest, dataset record, filename, provider output ve local artefaktları kapsar. Path resolver absolute path’i varsayılan olarak reddeder; traversal, symlink escape ve Windows reserved name kontrolleri uygular.

Transformers adapter varsayılan olarak `local_files_only=true`, `trust_remote_code=false` ve remote download kapalı çalışır. Manifest model root dışına çıkamaz; unknown field, unsafe capability, malformed checksum ve unsafe remote policy reddedilir. Resource limitleri message, prompt, output, config, manifest, report ve stream event boyutlarını sınırlar.

Secret detection yaygın API key, bearer, password, credential ve private-key biçimlerini hedefler; logging redaction uygular. Diagnostics yalnızca allowlist environment variable isimlerinin set durumunu gösterir, değerlerini göstermez. Bu önlemler OS sandbox ve bağımsız security review yerine geçmez.

