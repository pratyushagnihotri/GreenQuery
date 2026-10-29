from reasondb.interface.config import Config, get_default_configurator, operator_model_ids
from reasondb.operators.extract.text_qa_extract import TextQaExtract
from reasondb.operators.filter.extract_and_qa_filter import ExtractAndQaFilter
from reasondb.operators.filter.text_qa_filter import TextQaFilter

SMALL = "ollama/qwen2.5:3b-instruct"
LARGE = "ollama/qwen2.5:32b-instruct"


def text_ops(configurator, kind):
    toolbox = configurator.physical_operators
    pool = {
        TextQaFilter: toolbox.filter_operators,
        TextQaExtract: toolbox.extract_operators,
        ExtractAndQaFilter: toolbox.join_predicates,
    }[kind]
    return [op for op in pool if isinstance(op, kind)]


def test_single_model_mode_is_default(monkeypatch):
    monkeypatch.delenv("GREENQUERY_OPERATOR_MODELS", raising=False)
    assert operator_model_ids() == []
    configurator = get_default_configurator("api", openai_model_id=SMALL)
    for kind in (TextQaFilter, TextQaExtract, ExtractAndQaFilter):
        assert len(text_ops(configurator, kind)) == 1
    assert not hasattr(configurator, "multi_model_proxies")


def test_multi_model_mode_offers_one_operator_per_model(monkeypatch):
    monkeypatch.setenv("GREENQUERY_OPERATOR_MODELS", f"{SMALL}, {LARGE}")
    configurator = get_default_configurator("api", openai_model_id=LARGE)
    for kind in (TextQaFilter, TextQaExtract, ExtractAndQaFilter):
        ops = text_ops(configurator, kind)
        assert len(ops) == 2
        assert "qwen2.5:3b-instruct" in ops[0].get_operation_identifier()
        assert "qwen2.5:32b-instruct" in ops[1].get_operation_identifier()
        assert ops[0].quality < ops[1].quality
    proxies = configurator.multi_model_proxies
    assert len(proxies) == 3 and all("qwen2.5:3b-instruct" in proxy for proxy in proxies)


def test_lotus_uses_cheapest_model_as_proxy(monkeypatch):
    monkeypatch.setenv("GREENQUERY_OPERATOR_MODELS", f"{SMALL},{LARGE}")
    monkeypatch.setenv("GREENQUERY_OPTIMIZER", "lotus")
    executor = Config("test_operator_models", openai_model_id=LARGE).construct_executor()
    assert executor.optimizer.proxy_operators == set(executor.configurator.multi_model_proxies)
