from enum import Enum
import json
from typing import Any, Dict, List, Callable

# ------------------------------------------------------------
# 0. Common enums
# ------------------------------------------------------------
class Status(Enum):
    SUCCESS = 1
    FAILURE = 2
    RUNNING = 3


# ------------------------------------------------------------
# 1. Abstract node base class
# ------------------------------------------------------------
class BTNode:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec               # preserve the original description for visualization
        self.name = spec.get("name", spec["type"])
        self.children: List[BTNode] = []  # used by composite nodes

    # ---------- Runtime ----------
    def tick(self, ctx: Any) -> Status:
        raise NotImplementedError

    # ---------- Code generation ----------
    def to_code(self, indent: int = 0) -> str:
        """Convert the current node into Python code text"""
        raise NotImplementedError

    # ---------- Utilities ----------
    @staticmethod
    def make(spec: Dict[str, Any]) -> "BTNode":
        node_type = spec["type"]
        cls = {
            "Action": ActionNode,
            "Condition": ConditionNode,
            "Sequence": SequenceNode,
            "Selector": SelectorNode,
            "Loop": LoopNode,
            "ForEach": ForEachNode
        }.get(node_type)
        if cls is None:
            raise ValueError(f"Unknown node type: {node_type}")
        return cls(spec)


# ------------------------------------------------------------
# 2. Leaf nodes: Action & Condition
# ------------------------------------------------------------
class ActionNode(BTNode):
    def __init__(self, spec
                #  , api_map
                 ):
        super().__init__(spec)
        self.fn_name = spec["function"]
        self.args = spec.get("args", {})
        
    # ---------- Execute ----------
    def tick(self, ctx):
        pass
        return Status.SUCCESS

    # ---------- Code ----------
    def to_code(self, indent=0):
        pad = " " * (indent * 4)
        arg_list = []
        for k, v in self.args.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                var_name = v[2:-1]
                arg_list.append(f"{k}={var_name}")
            else:
                arg_list.append(f"{k}={repr(v)}")
        args_str = ", ".join(arg_list)
        return f"{pad}{self.fn_name}({args_str})\n"


class ConditionNode(BTNode):
    def __init__(self, spec):
        super().__init__(spec)
        self.fn_name = spec.get("function")
        self.args = spec.get("args", {})
        self.fn = None

        self.params = spec.get("params")
        self.expected = spec.get("expected")

        is_conditional_logic_present = self.fn_name or (self.params is not None)
        if not is_conditional_logic_present:
            raise ValueError(
                "ConditionNode must have 'function' or 'params'"
            )

    def _resolve_value(self, value: Any, ctx: Any) -> Any:
        if not (isinstance(value, str) and value.startswith("${") and value.endswith("}")):
            return value
        
        var_path = value[2:-1]
        parts = var_path.split('.')
        
        resolved_value = ctx.get(parts[0])
        if resolved_value is None:
            raise ValueError(f"Variable '{parts[0]}' not found in context")

        for part in parts[1:]:
            if isinstance(resolved_value, dict):
                resolved_value = resolved_value.get(part)
            else:
                resolved_value = getattr(resolved_value, part, None)
            
            if resolved_value is None:
                raise ValueError(f"Could not resolve '{part}' in '{var_path}'")
        return resolved_value

    def tick(self, ctx):
        condition_result = False
        if self.fn:
            resolved_args = {k: self._resolve_value(v, ctx) for k, v in self.args.items()}
            result = self.fn(ctx, **resolved_args)
        
            if self.expected is not None:
                expected_val = self._resolve_value(self.expected, ctx)
                condition_result = (result == expected_val)
            else:
                condition_result = bool(result)
        
        elif self.params is not None:
            param_val = self._resolve_value(self.params, ctx)
            expected_val = self._resolve_value(self.expected, ctx)
            condition_result = (param_val == expected_val)

        return Status.SUCCESS if condition_result else Status.FAILURE

    def to_code(self, indent=0):
        pad = " " * (indent * 4)

        def format_val(v):
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                return v[2:-1]
            return repr(v)

        condition_str = ""
        if self.fn_name:
            arg_list = []
            for k, v in self.args.items():
                arg_list.append(f"{k}={format_val(v)}")
            args_str = ", ".join(arg_list)
            left_str = f"{self.fn_name}({args_str})"
            
            if self.expected:
                condition_str = f"({left_str}) {self.expected}"
            else:
                condition_str = left_str
        elif self.params is not None:
            left = format_val(self.params)
            
            if isinstance(self.expected, str) and any(op in self.expected for op in ['==', '!=', '<', '>', ' is ', ' in ']):
                 right = self.expected
            else:
                 right = f"== {format_val(self.expected)}"
            condition_str = f"{left} {right}"
        else:
             return ""

        line = f"{pad}if not ({condition_str}):\n"
        line += f"{pad}    return False\n"
        return line


# ------------------------------------------------------------
# 3. Composite nodes: Sequence & Selector
# ------------------------------------------------------------
class SequenceNode(BTNode):
    def __init__(self, spec):
        super().__init__(spec)
        self.children = [BTNode.make(c) for c in spec["children"]]

    def tick(self, ctx):
        for ch in self.children:
            if ch.tick(ctx) != Status.SUCCESS:
                return Status.FAILURE
        return Status.SUCCESS

    def to_code(self, indent=0):
        code = ""
        for ch in self.children:
            code += ch.to_code(indent)
        code += "\n"
        return code


class SelectorNode(BTNode):
    def __init__(self, spec):
        super().__init__(spec)
        self.children = [BTNode.make(c) for c in spec["children"]]

    def tick(self, ctx):
        for ch in self.children:
            if ch.tick(ctx) == Status.SUCCESS:
                return Status.SUCCESS
        return Status.FAILURE

    def to_code(self, indent=0):
        pad = " " * (indent * 4)
        code = f"{pad}# Selector {self.name}\n"
        indent += 1
        child_func_list = []
        for i, ch in enumerate(self.children):
            child_func_name = f"{ch.name}"
            prefix = f"{pad}def {child_func_name}():\n"
            body = ch.to_code(indent)
            code += prefix + body + ""
            code += pad + " " * 4 + "return True\n"
            child_func_list.append(child_func_name)
            
        code += "\n"
        code += f"{pad}child_func_list = [" + ",".join(child_func_list) + "]\n"
        code += f"{pad}for child_func in child_func_list:\n"
        code += f"{pad}    if child_func():\n"
        code += f"{pad}        return True\n"
        code += "\n"
        return code
    

class LoopNode(BTNode):
    def __init__(self, spec):
        super().__init__(spec)
        self.child = BTNode.make(spec["child"])
        self.count = spec.get("count", None)
        self.until = spec.get("until", None)  # "SUCCESS" or "FAILURE"

    def tick(self, ctx):
        n = 0
        while self.count is None or n < self.count:
            status = self.child.tick(ctx)
            if self.until:
                if self.until == "SUCCESS" and status == Status.SUCCESS:
                    break
                if self.until == "FAILURE" and status == Status.FAILURE:
                    break
            n += 1
        return Status.SUCCESS

    def to_code(self, indent=0):
        pad = " " * (indent * 4)
        code = ""
        loop_head = ""
        if self.count is not None:
            loop_head = f"for _ in range({self.count}):\n"
        else:
            loop_head = "while True:\n"
        code += pad + loop_head
        code += self.child.to_code(indent + 1)
        if self.until:
            code += pad + " " * 4 + f"if child_status == {self.until}:\n"
            code += pad + " " * 8 + "break\n"
        code += "\n"
        return code
    

class ForEachNode(BTNode):
    def __init__(self, spec):
        super().__init__(spec)
        self.var = spec["var"]
        self.iterable = spec["iterable"]
        self.child = BTNode.make(spec["child"])

    def tick(self, ctx):
        iterable = ctx[self.iterable]
        for item in iterable:
            ctx[self.var] = item
            status = self.child.tick(ctx)
            if status != Status.SUCCESS:
                return status
        return Status.SUCCESS

    def to_code(self, indent=0):
        pad = " " * (indent * 4)
        code = f"{pad}for {self.var} in {self.iterable}:\n"
        code += self.child.to_code(indent + 1)
        code += "\n"
        return code
