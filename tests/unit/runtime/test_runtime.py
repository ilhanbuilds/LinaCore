import pytest
from lina_core.core.exceptions import ModelLoadError, ProviderNotFoundError, RuntimeNotReadyError, TokenizerError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.streaming import StreamEventType
from lina_core.models.metadata import ModelCapability
from lina_core.models.mock import MockModelProvider
from lina_core.runtime.health import HealthLevel
from lina_core.runtime.registry import ProviderRegistry
from lina_core.runtime.service import RuntimeService
from lina_core.tokenizer.mock import MockTokenizerProvider

def request(text:str="hello")->InferenceRequest:return InferenceRequest((ChatMessage(ChatRole.USER,text),))
def test_registry_list(): reg:ProviderRegistry[int]=ProviderRegistry(); reg.register("b",2); reg.register("a",1); assert reg.list()==("a","b")
def test_registry_duplicate():
    reg:ProviderRegistry[int]=ProviderRegistry(); reg.register("x",1)
    with pytest.raises(ValueError):reg.register("x",2)
def test_registry_missing():
    with pytest.raises(ProviderNotFoundError):ProviderRegistry[int]().get("x")
def test_mock_load_unload(): model=MockModelProvider(); model.load(); assert model.is_loaded; model.unload(); assert not model.is_loaded
def test_mock_requires_load():
    with pytest.raises(ModelLoadError):MockModelProvider().generate(request())
def test_mock_capabilities(): assert ModelCapability.STREAMING in MockModelProvider().metadata.capabilities
def test_runtime_inference(runtime:RuntimeService): assert runtime.infer(request()).text=="[mock] hello"
def test_runtime_deterministic(runtime:RuntimeService): assert runtime.infer(request()).text==runtime.infer(request()).text
def test_runtime_stream(runtime:RuntimeService):
    events=list(runtime.stream(request())); assert events[0].type is StreamEventType.STARTED and events[-1].type is StreamEventType.COMPLETED
def test_runtime_health(runtime:RuntimeService): assert runtime.health().status is HealthLevel.HEALTHY
def test_runtime_shutdown(runtime:RuntimeService): runtime.shutdown(); assert runtime.active_model_id is None
def test_runtime_not_ready():
    with pytest.raises(RuntimeNotReadyError):RuntimeService().infer(request())
@pytest.mark.parametrize("text",["","hello","Türkçe 💙","two words"])
def test_tokenizer_round_trip(text:str): tok=MockTokenizerProvider(); assert tok.decode(tok.encode(text))==text
def test_tokenizer_count(): assert MockTokenizerProvider().count_tokens("abc")==3
def test_tokenizer_invalid():
    with pytest.raises(TokenizerError):MockTokenizerProvider().decode([999])

