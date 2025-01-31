from python.xavier import (
    Array,
    OpName,
    OpType,
    Op,
    UnaryOp,
    BinaryOp,
    TransformOp,
)
from python.metal.context import MTLContext, MTLFusedKernel
from python.base import Compiler


class MTLCompiler(Compiler):
    _binary_ops = {
        OpName.ADD: "+",
        OpName.SUB: "-",
        OpName.MUL: "*",
        OpName.DIV: "/",
    }
    _unary_ops = {OpName.EXP: "metal::exp", OpName.LOG: "metal::log", OpName.NEG: "-"}

    def __init__(self, ctx: MTLContext):
        self._ctx = ctx
        self._terminals = {}
        self._symbol_table = {}

    def _recur_fuse(self, arr: Array) -> str:
        if arr.id() in self._symbol_table:
            return ""
        op: Op = arr.op()
        symbol = f"t{arr.id()}"
        self._symbol_table[arr.id()] = symbol
        match op.type():
            case OpType.INITIALIZER:
                return f"\tauto {symbol} = input{self._terminals[arr.id()]}[id];\n"
            case OpType.UNARY:
                unary_op: UnaryOp = op
                operand = unary_op.operand()
                s1 = self._recur_fuse(operand)
                s2 = f"\tauto {symbol} = {MTLCompiler._unary_ops[unary_op.name()]}({self._symbol_table[operand.id()]});\n"
                return s1 + s2
            case OpType.BINARY:
                binary_op: BinaryOp = op
                lhs = binary_op.lhs()
                rhs = binary_op.rhs()
                s1 = f"{self._recur_fuse(lhs)}{self._recur_fuse(rhs)}"
                s2 = f"\tauto {symbol} = {self._symbol_table[lhs.id()]} {MTLCompiler._binary_ops[binary_op.name()]} {self._symbol_table[rhs.id()]};\n"
                return s1 + s2

    def _fuse(self, arr: Array):
        fn_name = f"kernel{arr.id()}"
        src = f"""
#include <metal_stdlib>
template <class T>
[[kernel]] void {fn_name} (
"""
        for i in range(len(self._terminals)):
            src += f"\tdevice T *input{i}[[buffer({i})]],\n"
        src += f"\tdevice T *output[[buffer({len(self._terminals)})]],\n"
        src += "\tuint id [[thread_position_in_grid]]\n){\n"
        src += self._recur_fuse(arr)
        src += f"\toutput[id] = {self._symbol_table[arr.id()]};\n" + "}\n"
        src += f'template [[host_name("{fn_name}_{arr.dtype()}")]] [[kernel]] decltype({fn_name}<float>) {fn_name}<float>;\n'
        print(src)
        lib = self._ctx.device.newLibraryWithSource_options_error_(src, None, None)[0]
        fn = lib.newFunctionWithName_(f"{fn_name}_{arr.dtype()}")
        pipeline_state = self._ctx.device.newComputePipelineStateWithFunction_error_(fn, None)[0]
        kernel = MTLFusedKernel(src, pipeline_state, arr.dtype())
        self._ctx.register_kernel(fn_name, kernel)

    def _recur_fusable(self, arr: Array, visited: set) -> bool:
        if arr.id() in visited:
            return True
        visited.add(arr.id())
        op: Op = arr.op()
        match op.type():
            case OpType.INITIALIZER:
                # DO NOT add this to visited set
                # Initializers should not be marked as visited
                if arr.id() not in self._terminals:
                    self._terminals[arr.id()] = len(self._terminals)
                return True
            case OpType.UNARY:
                unary_op: UnaryOp = op
                return self._recur_fusable(unary_op.operand(), visited)
            case OpType.BINARY:
                binary_op: BinaryOp = op
                return self._recur_fusable(binary_op.lhs(), visited) and self._recur_fusable(binary_op.rhs(), visited)
        return False

    def _fusable(self, arr: Array) -> bool:
        self._terminals.clear()
        return self._recur_fusable(arr, set())

    def _recur_compile(self, arr: Array, visited: set):
        if arr.id() in visited:
            return
        visited.add(arr.id())
        if self._fusable(arr) and arr.op().type() != OpType.INITIALIZER:
            self._fuse(arr)
        else:
            op: Op = arr.op()
            match op.type():
                case OpType.UNARY:
                    unary_op: UnaryOp = op
                    self._recur_compile(unary_op.operand(), visited)
                case OpType.BINARY:
                    binary_op: BinaryOp = op
                    self._recur_compile(binary_op.lhs(), visited)
                    self._recur_compile(binary_op.rhs(), visited)
                case OpType.TRANSFORM:
                    transform_op: TransformOp = op
                    self._recur_compile(transform_op.operand(), visited)

    def compile(self, arr: Array):
        self._recur_compile(arr, set())
