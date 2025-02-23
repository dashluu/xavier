from python.xavier import (
    Array,
    Graph,
    Op,
    OpName,
    OpType,
    UnaryOp,
    BinaryOp,
    ConstOp,
    ArangeOp,
    BinaryOp,
    TransformOp,
    ReshapeOp,
    opnames,
)
from python.metal.kernels import constant, arange, ss_op, sparse_copy
from python.metal.context import MTLContext
from python.config import config


class MTLGraph(Graph):
    def __init__(self, root: Array):
        super().__init__(root)

    def _recur_find_terminals(self, arr: Array, visited: set, terminals: list[Array]):
        if arr.id() in visited:
            return
        visited.add(arr.id())
        op: Op = arr.op()
        match op.type():
            case OpType.INITIALIZER:
                self._initializer(arr)
                terminals.append(arr)
            case OpType.UNARY:
                unary_op: UnaryOp = op
                self._recur_find_terminals(unary_op.operand(), visited, terminals)
            case OpType.BINARY:
                binary_op: BinaryOp = op
                self._recur_find_terminals(binary_op.lhs(), visited, terminals)
                self._recur_find_terminals(binary_op.rhs(), visited, terminals)

    def _find_terminals(self, arr: Array) -> list[Array]:
        visited = set()
        terminals = []
        self._recur_find_terminals(arr, visited, terminals)
        return terminals

    def _call_kernel(self, kernel: str, arr: Array):
        terminals = self._find_terminals(arr)
        arr.alloc()
        ss_op(kernel, terminals, arr, config.ctx)

    def _initializer(self, arr: Array):
        arr.alloc()
        op: Op = arr.op()
        match op.name():
            case OpName.CONSTANT:
                const_op: ConstOp = op
                constant(arr, const_op.const(), config.ctx)
            case OpName.ARANGE:
                arange_op: ArangeOp = op
                arange(arr, arange_op.start(), arange_op.step(), config.ctx)

    def _unary(self, name: str, arr: Array, visited: set):
        unary_op: UnaryOp = arr.op()
        operand = unary_op.operand()
        self._recur_forw(operand, visited)
        arr.alloc()
        ss_op(name, [operand], arr, config.ctx)

    def _binary(self, name: str, arr: Array, visited: set):
        binary_op: BinaryOp = arr.op()
        lhs = binary_op.lhs()
        rhs = binary_op.rhs()
        self._recur_forw(lhs, visited)
        self._recur_forw(rhs, visited)
        arr.alloc()
        ss_op(name, [lhs, rhs], arr, config.ctx)

    def _transform(self, arr: Array):
        arr.alloc()
        op: Op = arr.op()
        match op.name():
            case OpName.RESHAPE:
                reshape_op: ReshapeOp = op
                operand = reshape_op.operand()
                if reshape_op.copy():
                    sparse_copy(f"sparse_copy", operand, arr, config.ctx)

    def _recur_forw(self, arr: Array, visited: set):
        if arr.id() in visited:
            return
        kernel = f"kernel{arr.id()}_{arr.dtype()}"
        ctx: MTLContext = config.ctx
        if kernel in ctx.kernels:
            # If the current operation is in the kernels, we can directly call it.
            self._call_kernel(kernel, arr)
        else:
            # If the current operation is not in the kernels, we need to evaluate it recursively.
            op: Op = arr.op()
            match op.type():
                case OpType.INITIALIZER:
                    self._initializer(arr)
                case OpType.UNARY:
                    self._unary(opnames[op.name()], arr, visited)
                case OpType.BINARY:
                    self._binary(opnames[op.name()], arr, visited)
                case OpType.TRANSFORM:
                    self._transform(arr)

    def compile(self):
        config.compiler.compile(self.root())

    def forward(self):
        visited = set()
        self._recur_forw(self.root(), visited)
