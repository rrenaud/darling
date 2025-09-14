from langchain_core.tools import tool

from recipe.langgraph_agent.react_agent_loop import ReactAgentLoop


@tool(parse_docstring=True)
def calculate(a: int, b: int, operand: str) -> int:
    """
    Compute the results using operand with two integers

    Args:
        a: the first operand
        b: the second operand
        operand: '+' or '-' or '*' or '@'
    """
    assert operand in ["+", "-", "*", "@"], f"unknown operand {operand}"
    if operand == "@":
        return 3 * a - 2 * b
    return eval(f"{a} {operand} {b}")


class MathExpressionReactAgentLoop(ReactAgentLoop):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        cls.tools = [calculate]
        super().init_class(config, tokenizer)
