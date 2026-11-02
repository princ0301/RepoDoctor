import time
from functools import wraps
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from repodoctor.deps import Deps
from repodoctor.nodes.explore import make_explore
from repodoctor.nodes.fix import make_fix
from repodoctor.nodes.setup import make_setup
from repodoctor.nodes.verify import make_verify
from repodoctor.state import RepoDoctorState
from repodoctor.nodes.reproduce import make_reproduce  


def _timed(name: str, fn):
    """Adds the node's wall-clock time to state['timings']."""
    @wraps(fn)
    def wrapper(state, *args, **kwargs):
        start = time.monotonic()
        out = fn(state, *args, **kwargs)
        timings = dict(state.get("timings", {}))
        timings[name] = round(timings.get(name, 0.0) + time.monotonic() - start, 1)
        return {**out, "timings": timings}

    return wrapper


def _after(next_node: str):
    def route(state: RepoDoctorState) -> str:
        return END if state.get("error") else next_node

    return route


def _after_verify(max_attempts: int):
    def route(state: RepoDoctorState) -> str:
        if state.get("success") or state.get("error"):
            return END
        return END if state.get("attempts", 0) >= max_attempts else "fix"

    return route


def build_graph(deps: Deps):
    """setup -> explore -> reproduce -> fix -> verify -> (fix again | end)"""
    g = StateGraph(RepoDoctorState)
    g.add_node("setup", _timed("setup", make_setup(deps)))
    g.add_node("explore", _timed("explore", make_explore(deps)))
    g.add_node("reproduce", _timed("reproduce", make_reproduce(deps)))
    g.add_node("fix", _timed("fix", make_fix(deps)))
    g.add_node("verify", _timed("verify", make_verify(deps)))

    g.add_edge(START, "setup")
    g.add_conditional_edges("setup", _after("explore"), {"explore": "explore", END: END})
    g.add_conditional_edges(
        "explore", _after("reproduce"), {"reproduce": "reproduce", END: END}
    )
    g.add_conditional_edges("reproduce", _after("fix"), {"fix": "fix", END: END})
    g.add_conditional_edges("fix", _after("verify"), {"verify": "verify", END: END})
    g.add_conditional_edges(
        "verify", _after_verify(deps.settings.max_attempts), {"fix": "fix", END: END}
    )
    return g.compile()


# --- placeholder graph from Phase 0 (used by `repodoctor hello`) ---

class HelloState(TypedDict, total=False):
    name: str
    greeting: str


def greet(state: HelloState) -> dict:
    return {"greeting": f"Hello, {state['name']}!"}


def build_hello_graph():
    g = StateGraph(HelloState)
    g.add_node("greet", greet)
    g.add_edge(START, "greet")
    g.add_edge("greet", END)
    return g.compile()
