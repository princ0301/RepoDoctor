from uuid import uuid4

from repodoctor.tracing import CallCounter


def test_call_counter_times_each_call_and_records_the_outcome():
    c = CallCounter()
    a, b = uuid4(), uuid4()
    c.on_chat_model_start({}, [[]], run_id=a, invocation_params={"model": "m1"})
    c.on_llm_end(None, run_id=a)
    c.on_chat_model_start({}, [[]], run_id=b, invocation_params={"model_name": "m2"})
    c.on_llm_error(RuntimeError("x"), run_id=b)
    assert c.calls == 2
    assert [(e["model"], e["ok"]) for e in c.timeline] == [("m1", True), ("m2", False)]
