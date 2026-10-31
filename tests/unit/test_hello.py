from repodoctor.graph import build_hello_graph


def test_hello_graph():
    assert build_hello_graph().invoke({"name": "x"})["greeting"] == "Hello, x!"