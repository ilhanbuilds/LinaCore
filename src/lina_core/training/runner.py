import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from lina_core.core.exceptions import TrainingError
from lina_core.datasets.sft import (
    SFTRecord,
    dataset_fingerprint,
    load_sft_dataset,
    split_sft_dataset,
)
from lina_core.datasets.validation import load_manifest
from lina_core.inference.chat_template import ChatTemplateRenderer
from lina_core.security.paths import safe_resolve
from lina_core.training.config import TrainingConfig

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
BASE_LICENSE = "Apache-2.0"
LINA_MODEL_ID = "lina-135m-sft-v0.1"
LINA_MODEL_NAME = "Lina-135M-SFT-v0.1"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    run_name: str
    base_model: str
    base_revision: str
    dataset: str
    dataset_fingerprint: str
    record_count: int
    device: str
    dtype: str
    method: str
    batch_size: int
    epochs: int
    max_steps: int
    output_dir: str
    download_required: bool
    trust_remote_code: bool
    report_to: str
    stages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model_id: str
    artifact_dir: str
    dataset_fingerprint: str
    duration_seconds: float
    global_steps: int
    epochs: int
    final_training_loss: float
    validation_loss: float | None
    perplexity: float | None
    manifest_path: str
    model_card_path: str
    metadata_path: str


class TrainingRunner:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def plan(self, config: TrainingConfig) -> TrainingPlan:
        if config.resume_from_checkpoint is not None:
            raise TrainingError(
                "Ara checkpoint resume henüz desteklenmiyor; yeni run başlatılmadı."
            )
        dataset_path = safe_resolve(self.workspace, config.dataset_path)
        output = safe_resolve(self.workspace, config.output_dir)
        records = load_sft_dataset(dataset_path)
        base_path = Path(config.base_model)
        download_required = (
            not (base_path.is_absolute() and base_path.exists())
            and not safe_resolve(self.workspace, config.base_model).exists()
        )
        if download_required and (config.local_files_only or not config.allow_download):
            raise TrainingError("Base model local değil; açık download izni verilmedi.")
        return TrainingPlan(
            config.run_name,
            BASE_MODEL_ID,
            config.base_revision,
            dataset_path.name,
            dataset_fingerprint(records),
            len(records),
            config.device,
            config.dtype,
            "full-supervised-fine-tuning",
            config.batch_size,
            config.num_train_epochs,
            config.max_steps,
            output.as_posix(),
            download_required,
            config.trust_remote_code,
            config.report_to,
            (
                "config-validation",
                "dataset-validation",
                "tokenization",
                "training",
                "validation",
                "artifact-save",
                "metadata",
            ),
        )

    def dry_run(self, config: TrainingConfig) -> dict[str, Any]:
        if config.dataset_manifest and config.dataset_path.endswith(".json"):
            manifest = safe_resolve(self.workspace, config.dataset_path)
            output = safe_resolve(self.workspace, config.output_dir)
            load_manifest(manifest)
            return {
                "status": "dry-run-valid",
                "run_name": config.run_name,
                "model_id": config.base_model,
                "output_dir": str(output),
                "environment": {
                    "python": sys.version.split()[0],
                    "platform": platform.system(),
                },
                "training_started": False,
            }
        return {**asdict(self.plan(config)), "status": "plan-valid", "training_started": False}

    def run(self, config: TrainingConfig) -> TrainingResult:
        plan = self.plan(config)
        output = safe_resolve(self.workspace, config.output_dir)
        self.validate_output(config)
        temporary = output.parent / f".{output.name}.incomplete-{uuid4().hex[:8]}"
        if temporary.exists():
            raise TrainingError("Geçici training dizini zaten mevcut.")
        temporary.mkdir(parents=True)
        started = time.perf_counter()
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("DISABLE_TELEMETRY", "1")
        try:
            result = self._train(config, plan, temporary, started)
            if output.exists():
                if not config.overwrite_output:
                    raise TrainingError("Output dizini mevcut; overwrite izni yok.")
                shutil.rmtree(output)
            temporary.replace(output)
            return TrainingResult(
                result.model_id,
                output.as_posix(),
                result.dataset_fingerprint,
                result.duration_seconds,
                result.global_steps,
                result.epochs,
                result.final_training_loss,
                result.validation_loss,
                result.perplexity,
                (output / "linacore-model.toml").as_posix(),
                (output / "MODEL_CARD.md").as_posix(),
                (output / "training-metadata.json").as_posix(),
            )
        except Exception:
            (temporary / "FAILED").write_text("Training tamamlanmadı.\n", encoding="utf-8")
            raise

    def validate_output(self, config: TrainingConfig) -> Path:
        output = safe_resolve(self.workspace, config.output_dir)
        if output.exists() and any(output.iterdir()) and not config.overwrite_output:
            raise TrainingError("Output dizini dolu; overwrite_output açık değil.")
        return output

    def _train(
        self, config: TrainingConfig, plan: TrainingPlan, temporary: Path, started: float
    ) -> TrainingResult:
        try:
            import numpy as np
            import torch
            import transformers
        except ImportError as exc:
            raise TrainingError("Training optional dependency grubu kurulu değil.") from exc
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        dataset_path = safe_resolve(self.workspace, config.dataset_path)
        records = load_sft_dataset(dataset_path)
        split = split_sft_dataset(
            records,
            seed=config.seed,
            train_ratio=config.train_split,
            validation_ratio=config.validation_split,
            test_ratio=config.test_split,
        )
        base_path = safe_resolve(self.workspace, config.base_model)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_path,
            revision=config.base_revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise TrainingError("Tokenizer pad veya eos token içermiyor.")
            tokenizer.pad_token = tokenizer.eos_token
        dtype: Any = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.dtype]
        dtype_keyword = (
            "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        )
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(
            base_path,
            revision=config.base_revision,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
            **{dtype_keyword: dtype},
        )
        model.to(config.device)
        model.train()
        if config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train_items = list(split.train)
        losses: list[float] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        for _epoch in range(config.num_train_epochs):
            for start_index in range(0, len(train_items), config.batch_size):
                batch = train_items[start_index : start_index + config.batch_size]
                encoded = self._encode(tokenizer, batch, config.max_sequence_length, torch)
                outputs = model(**encoded)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()
                losses.append(float(outputs.loss.detach().cpu()))
                if (global_step + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if config.max_steps and global_step >= config.max_steps:
                    break
            if config.max_steps and global_step >= config.max_steps:
                break
        if not losses:
            raise TrainingError("Training step üretilmedi.")
        validation_loss = self._validation_loss(
            model, tokenizer, list(split.validation), config.max_sequence_length, torch
        )
        model_dir = temporary / "model"
        tokenizer_dir = temporary / "tokenizer"
        report_dir = temporary / "reports"
        model_dir.mkdir()
        tokenizer_dir.mkdir()
        report_dir.mkdir()
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(tokenizer_dir)
        duration = time.perf_counter() - started
        fingerprint = dataset_fingerprint(records)
        perplexity = (
            math.exp(validation_loss)
            if validation_loss is not None and validation_loss < 20
            else None
        )
        metadata = {
            "model_name": LINA_MODEL_NAME,
            "model_id": LINA_MODEL_ID,
            "lina_model_version": "0.1",
            "base_model": BASE_MODEL_ID,
            "base_revision": config.base_revision,
            "base_license": BASE_LICENSE,
            "training_method": "Supervised Fine-Tuning (assistant-only selective causal LM loss)",
            "dataset_version": "0.1",
            "dataset_fingerprint": fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "numpy_version": np.__version__,
            "device": config.device,
            "dtype": config.dtype,
            "seed": config.seed,
            "epochs": config.num_train_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "max_sequence_length": config.max_sequence_length,
            "training_steps": global_step,
            "final_training_loss": losses[-1],
            "validation_loss": validation_loss,
            "perplexity": perplexity,
            "artifact_format": "Transformers Safetensors merged full model",
            "adapter": False,
            "gguf_export_status": "not-exported",
            "platform": platform.system(),
            "config_fingerprint": config.fingerprint,
            "known_limitations": [
                "Küçük sentetik dataset ve kısa eğitim nedeniyle kalite deneysel düzeydedir.",
                "Model sıfırdan pretrained değildir; SmolLM2-135M türevidir.",
            ],
        }
        (temporary / "training-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (report_dir / "training-report.json").write_text(
            json.dumps({"losses": losses, **metadata}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (report_dir / "checkpoint-metadata.json").write_text(
            json.dumps(
                {
                    "global_step": global_step,
                    "epoch": config.num_train_epochs,
                    "optimizer_state": "not-saved",
                    "scheduler_state": "not-used",
                    "rng_state": "seed-reproducible",
                    "config_fingerprint": config.fingerprint,
                    "dataset_fingerprint": fingerprint,
                    "base_revision": config.base_revision,
                    "resumable": False,
                    "limitation": "Ara checkpoint resume bu sürümde henüz desteklenmiyor.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_manifest(temporary, config, fingerprint)
        self._write_model_card(temporary, metadata, plan.record_count)
        checksums: dict[str, str] = {}
        for artifact in sorted(path for path in temporary.rglob("*") if path.is_file()):
            if artifact.name == "checksums.json":
                continue
            checksums[artifact.relative_to(temporary).as_posix()] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
        (temporary / "checksums.json").write_text(
            json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return TrainingResult(
            LINA_MODEL_ID,
            temporary.as_posix(),
            fingerprint,
            duration,
            global_step,
            config.num_train_epochs,
            losses[-1],
            validation_loss,
            perplexity,
            "linacore-model.toml",
            "MODEL_CARD.md",
            "training-metadata.json",
        )

    @staticmethod
    def _encode(
        tokenizer: Any, records: list[SFTRecord], max_length: int, torch: Any
    ) -> dict[str, Any]:
        renderer = ChatTemplateRenderer(fallback_enabled=True)
        texts = [
            renderer.render(record.messages, add_generation_prompt=False).text
            + (tokenizer.eos_token or "")
            for record in records
        ]
        prompts = [
            renderer.render(record.messages[:-1], add_generation_prompt=True).text
            for record in records
        ]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100
        for index, prompt in enumerate(prompts):
            prompt_length = len(
                tokenizer(prompt, truncation=True, max_length=max_length)["input_ids"]
            )
            labels[index, :prompt_length] = -100
        return {**encoded, "labels": labels}

    @classmethod
    def _validation_loss(
        cls, model: Any, tokenizer: Any, records: list[SFTRecord], max_length: int, torch: Any
    ) -> float | None:
        if not records:
            return None
        model.eval()
        values: list[float] = []
        with torch.no_grad():
            for record in records:
                outputs = model(**cls._encode(tokenizer, [record], max_length, torch))
                values.append(float(outputs.loss.detach().cpu()))
        model.train()
        return sum(values) / len(values)

    @staticmethod
    def _write_manifest(root: Path, config: TrainingConfig, fingerprint: str) -> None:
        content = f'''schema_version = 1
model_id = "{LINA_MODEL_ID}"
display_name = "{LINA_MODEL_NAME}"
version = "0.1"
backend = "transformers"
architecture = "SmolLM2ForCausalLM"
model_path = "model"
tokenizer_path = "tokenizer"
source = "derived-from:{BASE_MODEL_ID}"
source_revision = "{config.base_revision}"
license = "Apache-2.0"
context_length = {config.max_sequence_length}
default_device = "{config.device}"
default_dtype = "{config.dtype}"
quantization = "none"
capabilities = ["text_generation", "streaming"]
chat_template_policy = "fallback"
trust_remote_code = false
local_files_only = true
parameter_count = 135000000
notes = "Lina SFT v0.1; dataset {fingerprint}; assistant-only selective loss."
'''
        (root / "linacore-model.toml").write_text(content, encoding="utf-8")

    @staticmethod
    def _write_model_card(root: Path, metadata: dict[str, Any], record_count: int) -> None:
        text = f"""# {LINA_MODEL_NAME}

## Model özeti

Bu model sıfırdan pretrained bir foundation model değildir. Apache-2.0 lisanslı
`{BASE_MODEL_ID}` revision `{metadata["base_revision"]}` üzerine Lina'ya özel sentetik
veriyle supervised fine-tuning uygulanmış deneysel bir türevdir.

## Amaçlanan kullanım

Local LinaCore inference, eğitim pipeline doğrulaması ve Türkçe asistan davranışı araştırması.
Yüksek riskli kararlar, güvenlik garantisi veya doğrulanmamış uzmanlık için amaçlanmamıştır.

## Eğitim

- Yöntem: {metadata["training_method"]}
- Dataset: Lina SFT v0.1, {record_count} sentetik kayıt, CC0-1.0
- Fingerprint: `{metadata["dataset_fingerprint"]}`
- Donanım: {metadata["device"]}, {metadata["dtype"]}
- Seed/step: {metadata["seed"]} / {metadata["training_steps"]}
- Final loss: {metadata["final_training_loss"]}
- Validation loss: {metadata["validation_loss"]}

## Evaluation ve base model karşılaştırması

Training sırasında validation loss ve perplexity hesaplanır. Dokuz davranış promptu için
base/fine-tuned karşılaştırması `scripts/compare_local_models.py` ile üretilir ve
`reports/base-vs-lina.json` altında saklanır. Keyword sonucu insan değerlendirmesi değildir;
`human_review` alanı bilinçli olarak boş bırakılır.

## Lisans ve yükümlülükler

Base model Apache-2.0 lisanslıdır. Türetilmiş ağırlık dağıtılırken Apache-2.0 lisans ve
notice yükümlülükleri korunmalı, yapılan değişiklik belirtilmeli ve üçüncü taraf hakları
ayrıca değerlendirilmelidir. Dataset örneği CC0-1.0'dır.

## Sınırlamalar, güvenlik ve gizlilik

Küçük sentetik dataset ve kısa eğitim nedeniyle kalite düşebilir; doğru veya güvenli cevap
garantisi yoktur. Kullanıcıya özel veriyle eğitilmemiştir. Secret ve kişisel veri prompt veya
training loglarına konulmamalıdır. Loss yalnız son assistant cevabının tokenları üzerinde
hesaplanmıştır.

## LinaCore ile çalıştırma

```console
lina-core inference generate --model {LINA_MODEL_ID} --prompt "Merhaba, sen kimsin?"
```

GGUF henüz export edilmemiştir. Reproducibility ayrıntıları `training-metadata.json` içindedir.
"""
        (root / "MODEL_CARD.md").write_text(text, encoding="utf-8")
