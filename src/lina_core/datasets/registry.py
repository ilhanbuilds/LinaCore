from lina_core.datasets.contracts import DatasetManifest
from lina_core.runtime.registry import ProviderRegistry


class DatasetRegistry(ProviderRegistry[DatasetManifest]):
    def add(self, manifest: DatasetManifest) -> None:
        self.register(manifest.dataset_id, manifest)
