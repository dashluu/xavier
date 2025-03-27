from python.xavier import Array, MTLGraph, MTLContext
import numpy as np
import torch


def compare_grads(arr_grad: Array, torch_grad: torch.Tensor, name: str):
    t_grad = torch.frombuffer(arr_grad, dtype=torch.float32)
    assert torch.allclose(t_grad, torch_grad.flatten(), atol=1e-3, rtol=0), f"Gradient mismatch for {name}"


class TestBackprop:
    lib = "./xavier/build/backend/metal/kernels.metallib"

    def test_backprop_v1(self):
        ctx = MTLContext(self.lib)
        print("backprop 1:")
        n = np.random.randint(1, 5)
        shape = [np.random.randint(1, 100) for _ in range(n)]
        np1 = np.random.randn(*shape).astype(np.float32)
        np2 = np.random.randn(*shape).astype(np.float32)
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)
        arr3 = arr1 + arr2
        arr4 = arr1 * arr2
        arr5 = arr3 + arr4
        arr6 = arr3 * arr4
        arr7 = arr5 + arr6
        g = MTLGraph(arr7, ctx)
        g.compile()
        g.forward()
        g.backward()
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)
        t3 = t1 + t2
        t3.retain_grad()
        t4 = t1 * t2
        t4.retain_grad()
        t5 = t3 + t4
        t5.retain_grad()
        t6 = t3 * t4
        t6.retain_grad()
        t7 = t5 + t6
        t7.retain_grad()
        t7.sum().backward()
        t8 = torch.frombuffer(arr7, dtype=torch.float32)
        assert torch.allclose(t8, t7.flatten(), atol=1e-3, rtol=0)
        t9 = torch.frombuffer(arr3.grad, dtype=torch.float32)
        assert torch.allclose(t9, t3.grad.flatten(), atol=1e-3, rtol=0)
        compare_grads(arr3.grad, t3.grad, "arr3")
        compare_grads(arr4.grad, t4.grad, "arr4")
        compare_grads(arr5.grad, t5.grad, "arr5")
        compare_grads(arr6.grad, t6.grad, "arr6")
        compare_grads(arr7.grad, t7.grad, "arr7")

    def test_backprop_v2(self):
        ctx = MTLContext(self.lib)
        print("Testing complex unary(and one binary) operations chain:")

        shape = [2, 3]
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)  # Positive values for log

        # Xavier implementation: log(exp(x) * x) / x
        arr1 = Array.from_numpy(np1)
        arr2 = arr1.exp()
        arr3 = arr2 * arr1
        arr4 = arr3.log()
        arr5 = arr4 / arr1

        g = MTLGraph(arr5, ctx)
        g.compile()
        g.forward()
        g.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.exp(t1)
        t2.retain_grad()
        t3 = t2 * t1
        t3.retain_grad()
        t4 = torch.log(t3)
        t4.retain_grad()
        t5 = t4 / t1
        t5.sum().backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "input")
        compare_grads(arr2.grad, t2.grad, "exp")
        compare_grads(arr3.grad, t3.grad, "mul")
        compare_grads(arr4.grad, t4.grad, "log")

    def test_backprop_v3(self):
        ctx = MTLContext(self.lib)
        print("Testing branched operations:")

        shape = [3, 4, 2]
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # Xavier implementation
        # Branch 1: log(x1) * exp(x2)
        # Branch 2: x1 / x2
        # Result: Branch1 + Branch2
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)

        # Branch 1
        b1_1 = arr1.log()
        b1_2 = arr2.exp()
        branch1 = b1_1 * b1_2

        # Branch 2
        branch2 = arr1 / arr2

        # Combine branches
        result = branch1 + branch2

        g = MTLGraph(result, ctx)
        g.compile()
        g.forward()
        g.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)

        # Branch 1
        tb1_1 = torch.log(t1)
        tb1_1.retain_grad()
        tb1_2 = torch.exp(t2)
        tb1_2.retain_grad()
        tbranch1 = tb1_1 * tb1_2
        tbranch1.retain_grad()

        # Branch 2
        tbranch2 = t1 / t2
        tbranch2.retain_grad()

        # Combine branches
        tresult = tbranch1 + tbranch2
        tresult.sum().backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "input1")
        compare_grads(arr2.grad, t2.grad, "input2")
        compare_grads(b1_1.grad, tb1_1.grad, "log")
        compare_grads(b1_2.grad, tb1_2.grad, "exp")
        compare_grads(branch1.grad, tbranch1.grad, "branch1")
        compare_grads(branch2.grad, tbranch2.grad, "branch2")

    def test_backprop_v4(self):
        ctx = MTLContext(self.lib)
        print("Testing nested operations:")

        shape = [5, 7, 2, 4]
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # Xavier implementation: log(exp(x1/x2) * recip(x1))
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)

        div1 = arr1 / arr2
        exp1 = div1.exp()
        rec1 = arr1.recip()
        mul1 = exp1 * rec1
        result = mul1.log()

        g = MTLGraph(result, ctx)
        g.compile()
        g.forward()
        g.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)

        tdiv1 = t1 / t2
        tdiv1.retain_grad()
        texp1 = torch.exp(tdiv1)
        texp1.retain_grad()
        trec1 = 1.0 / t1
        trec1.retain_grad()
        tmul1 = texp1 * trec1
        tmul1.retain_grad()
        tresult = torch.log(tmul1)
        tresult.sum().backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "input1")
        compare_grads(arr2.grad, t2.grad, "input2")
        compare_grads(div1.grad, tdiv1.grad, "div")
        compare_grads(exp1.grad, texp1.grad, "exp")
        compare_grads(rec1.grad, trec1.grad, "recip")
        compare_grads(mul1.grad, tmul1.grad, "mul")

    def test_backprop_v5(self):
        ctx = MTLContext(self.lib)
        print("Testing square and sqrt operations:")

        shape = [3, 4, 2]
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # Xavier implementation: sqrt(x^2) + x^2/sqrt(x)
        arr1 = Array.from_numpy(np1)

        # Branch 1: sqrt(x^2)
        sq1 = arr1.sq()
        sqrt1 = sq1.sqrt()

        # Branch 2: x^2/sqrt(x)
        sq2 = arr1.sq()
        sqrt2 = arr1.sqrt()
        div1 = sq2 / sqrt2

        # Combine branches
        result = sqrt1 + div1

        g = MTLGraph(result, ctx)
        g.compile()
        g.forward()
        g.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)

        # Branch 1: sqrt(x^2)
        tsq1 = t1 * t1
        tsq1.retain_grad()
        tsqrt1 = torch.sqrt(tsq1)
        tsqrt1.retain_grad()

        # Branch 2: x^2/sqrt(x)
        tsq2 = t1 * t1
        tsq2.retain_grad()
        tsqrt2 = torch.sqrt(t1)
        tsqrt2.retain_grad()
        tdiv1 = tsq2 / tsqrt2
        tdiv1.retain_grad()

        # Combine branches
        tresult = tsqrt1 + tdiv1
        tresult.sum().backward()

        # Compare gradients
        compare_grads(arr1.grad, t1.grad, "input")
        compare_grads(sq1.grad, tsq1.grad, "square1")
        compare_grads(sqrt1.grad, tsqrt1.grad, "sqrt1")
        compare_grads(sq2.grad, tsq2.grad, "square2")
        compare_grads(sqrt2.grad, tsqrt2.grad, "sqrt2")
        compare_grads(div1.grad, tdiv1.grad, "div")

    def test_backprop_twice(self):
        ctx = MTLContext(self.lib)
        print("Testing double backpropagation with complex operations:")

        shape = [2, 3, 4]
        np1 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)
        np2 = np.random.uniform(0.1, 2.0, size=shape).astype(np.float32)

        # Xavier implementation
        # f(x1, x2) = log(sqrt(x1^2) * exp(x2/x1)) + (x1 * sqrt(x2))^2
        arr1 = Array.from_numpy(np1)
        arr2 = Array.from_numpy(np2)

        # Branch 1: log(sqrt(x1^2) * exp(x2/x1))
        sq1 = arr1.sq()
        sqrt1 = sq1.sqrt()
        div1 = arr2 / arr1
        exp1 = div1.exp()
        mul1 = sqrt1 * exp1
        log1 = mul1.log()

        # Branch 2: (x1 * sqrt(x2))^2
        sqrt2 = arr2.sqrt()
        mul2 = arr1 * sqrt2
        sq2 = mul2.sq()

        # Combine branches
        result = log1 + sq2

        # First backward pass
        g = MTLGraph(result, ctx)
        g.compile()
        g.forward()
        g.backward()

        # PyTorch implementation
        t1 = torch.from_numpy(np1).requires_grad_(True)
        t2 = torch.from_numpy(np2).requires_grad_(True)

        # Branch 1
        tsq1 = t1 * t1
        tsq1.retain_grad()
        tsqrt1 = torch.sqrt(tsq1)
        tsqrt1.retain_grad()
        tdiv1 = t2 / t1
        tdiv1.retain_grad()
        texp1 = torch.exp(tdiv1)
        texp1.retain_grad()
        tmul1 = tsqrt1 * texp1
        tmul1.retain_grad()
        tlog1 = torch.log(tmul1)
        tlog1.retain_grad()

        # Branch 2
        tsqrt2 = torch.sqrt(t2)
        tsqrt2.retain_grad()
        tmul2 = t1 * tsqrt2
        tmul2.retain_grad()
        tsq2 = tmul2 * tmul2
        tsq2.retain_grad()

        # Combine branches
        tresult = tlog1 + tsq2
        tsum = tresult.sum()
        tsum.backward(retain_graph=True)

        # Compare first backward pass gradients
        print("\nChecking first backward pass:")
        compare_grads(arr1.grad, t1.grad, "input1")
        compare_grads(arr2.grad, t2.grad, "input2")
        compare_grads(sq1.grad, tsq1.grad, "square1")
        compare_grads(sqrt1.grad, tsqrt1.grad, "sqrt1")
        compare_grads(div1.grad, tdiv1.grad, "div")
        compare_grads(exp1.grad, texp1.grad, "exp")
        compare_grads(mul1.grad, tmul1.grad, "mul1")
        compare_grads(log1.grad, tlog1.grad, "log")
        compare_grads(sqrt2.grad, tsqrt2.grad, "sqrt2")
        compare_grads(mul2.grad, tmul2.grad, "mul2")
        compare_grads(sq2.grad, tsq2.grad, "square2")

        # Modify the result slightly and run backward again
        # TODO: uncomment this after implementing backprop for broadcasting
        # result = result * 2.0
        # g = MTLGraph(result, ctx)
        # g.compile()
        # g.forward()
        g.backward()

        # PyTorch second pass
        # TODO: uncomment this after implementing backprop for broadcasting
        # tresult = tresult * 2.0
        # Clear all gradients from PyTorch computation graph
        t1.grad = None
        t2.grad = None
        tsq1.grad = None
        tsqrt1.grad = None
        tdiv1.grad = None
        texp1.grad = None
        tmul1.grad = None
        tlog1.grad = None
        tsqrt2.grad = None
        tmul2.grad = None
        tsq2.grad = None
        tresult.grad = None
        tsum.backward()

        # Compare second backward pass gradients
        print("\nChecking second backward pass:")
        compare_grads(arr1.grad, t1.grad, "input1 (2nd pass)")
        compare_grads(arr2.grad, t2.grad, "input2 (2nd pass)")
        compare_grads(sq1.grad, tsq1.grad, "square1 (2nd pass)")
        compare_grads(sqrt1.grad, tsqrt1.grad, "sqrt1 (2nd pass)")
        compare_grads(div1.grad, tdiv1.grad, "div (2nd pass)")
        compare_grads(exp1.grad, texp1.grad, "exp (2nd pass)")
        compare_grads(mul1.grad, tmul1.grad, "mul1 (2nd pass)")
        compare_grads(log1.grad, tlog1.grad, "log (2nd pass)")
        compare_grads(sqrt2.grad, tsqrt2.grad, "sqrt2 (2nd pass)")
        compare_grads(mul2.grad, tmul2.grad, "mul2 (2nd pass)")
        compare_grads(sq2.grad, tsq2.grad, "square2 (2nd pass)")
