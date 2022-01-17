# Thanks to yamnikov-oleg on GitHub
# https://blog.oyam.dev/python-formulas/

import ast
import operator
from typing import Any, Dict


def byte_offset_to_char_offset(source: str, byte_offset: int) -> int:
    while True:
        try:
            pre_source = source.encode()[:byte_offset].decode()
            break
        except UnicodeDecodeError:
            byte_offset -= 1
            continue
    return len(pre_source)


class FormulaError(Exception):
    pass


class FormulaSyntaxError(FormulaError):
    def __init__(self, msg: str, lineno: int, offset: int):
        self.msg = msg
        self.lineno = lineno
        self.offset = offset

    @classmethod
    def from_ast_node(cls, source: str, node: ast.AST, msg: str) -> "FormulaSyntaxError":
        lineno = node.lineno
        col_offset = node.col_offset
        offset = byte_offset_to_char_offset(source, col_offset)
        return cls(msg=msg, lineno=lineno, offset=offset + 1)

    @classmethod
    def from_syntax_error(cls, error: SyntaxError, msg: str) -> "FormulaSyntaxError":
        return cls(msg=f"{msg}: {error.msg}", lineno=error.lineno, offset=error.offset)

    def __str__(self):
        return f"{self.lineno}:{self.offset}: {self.msg}"


class FormulaRuntimeError(FormulaError):
    pass


MAX_FORMULA_LENGTH = 255


def evaluate_formula(formula: str, vars: Dict[str, Any]) -> float:
    if len(formula) > MAX_FORMULA_LENGTH:
        raise FormulaSyntaxError("The formula is too long", 1, 1)

    try:
        node = ast.parse(formula, "<string>", mode="eval")
    except SyntaxError as e:
        raise FormulaSyntaxError.from_syntax_error(e, "Could not parse")

    try:
        return eval_node(formula, node, vars)
    except FormulaSyntaxError:
        raise
    except Exception as e:
        raise FormulaRuntimeError(f"Evaluation failed: {e}")


def eval_node(source: str, node: ast.AST, vars: Dict[str, Any]) -> float:
    EVALUATORS = {
        ast.Expression: eval_expression,
        ast.Constant: eval_constant,
        ast.Name: eval_name,
        ast.BinOp: eval_binop,
        ast.UnaryOp: eval_unaryop,
    }

    for ast_type, evaluator in EVALUATORS.items():
        if isinstance(node, ast_type):
            return evaluator(source, node, vars)

    raise FormulaSyntaxError.from_ast_node(source, node, "This syntax is not supported")


def eval_expression(source: str, node: ast.Expression, vars: Dict[str, Any]) -> float:
    return eval_node(source, node.body, vars)


def eval_constant(source: str, node: ast.Constant, vars: Dict[str, Any]) -> float:
    if isinstance(node.value, int) or isinstance(node.value, float):
        return float(node.value)
    else:
        raise FormulaSyntaxError.from_ast_node(source, node, "Literals of this type are not supported")


def eval_name(source: str, node: ast.Name, vars: Dict[str, Any]) -> float:
    try:
        return float(vars[node.id])
    except KeyError:
        raise FormulaSyntaxError.from_ast_node(source, node, f"Undefined variable: {node.id}")


def eval_binop(source: str, node: ast.BinOp, vars: Dict[str, Any]) -> float:
    OPERATIONS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    left_value = eval_node(source, node.left, vars)
    right_value = eval_node(source, node.right, vars)

    try:
        apply = OPERATIONS[type(node.op)]
    except KeyError:
        raise FormulaSyntaxError.from_ast_node(source, node, "Operations of this type are not supported")

    return apply(left_value, right_value)


def eval_unaryop(source: str, node: ast.UnaryOp, vars: Dict[str, Any]) -> float:
    OPERATIONS = {
        ast.USub: operator.neg,
    }

    operand_value = eval_node(source, node.operand, vars)

    try:
        apply = OPERATIONS[type(node.op)]
    except KeyError:
        raise FormulaSyntaxError.from_ast_node(source, node, "Operations of this type are not supported")

    return apply(operand_value)
