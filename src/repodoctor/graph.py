from typing import TypedDict
from langgraph.graph import END, START, StateGraph

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