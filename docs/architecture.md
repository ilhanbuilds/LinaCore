# Architecture

Dependencies point inward: CLI/scripts and future transports call application services; services use domain contracts; providers implement those contracts. `core` and `security` are shared foundations and must not import Desktop or transport code.

The runtime owns lifecycle, provider registries and active model/tokenizer selection. Inference request/response/stream event objects are transport-neutral. Models and tokenizers implement abstract interfaces; the mock implementations validate wiring only. Serving maps controlled results to typed API responses and never exposes raw exceptions. Dataset validation feeds future training and evaluation, while training remains an orchestration boundary rather than a framework choice. Evaluation consumes runtime contracts and writes bounded local reports. Security applies at path, input, resource and logging boundaries.

