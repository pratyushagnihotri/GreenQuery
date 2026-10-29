from reasondb.database.indentifier import DataType
from reasondb.operators.filter.traditional_filter import TraditionalFilter


def test_traditional_filter_excludes_images():
    interface = TraditionalFilter(quality=1, fake_cost=0).get_llm_parameters()
    assert interface.is_applicable_to_input_datatypes(frozenset({DataType.BOOL}))
    assert not interface.is_applicable_to_input_datatypes(frozenset({DataType.IMAGE}))
