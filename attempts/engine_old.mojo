"""ISSUES
- Operations are working fine (add, mul, etc)
- Issue with Gradient 

- backward is not working 
- return out -> this returns a copy of the out Value
- Will have to return the original using Reference
- Getting error when using Reference `error for lifetime`
"""

from collections import Set, List

# @value
# struct Test:
#     var a: Int

# struct TestValue:
#     """Stores a single scalar value and its gradient."""
    
#     var data: Float32
#     var grad: Float32
#     var _backward: fn () -> None
#     # var _prev: Set[Value]
#     var _op: String

#     # fn __init__(inout self, data: Float32, _children: List[Value] = List[Value](), _op: String = ""):
#     fn __init__(inout self, data: Float32, _backward: fn () -> None, _op: String = ""):
#         self.data = data
#         self.grad = Float32(0)

#         # internal variables used for autograd graph constrution
#         self._backward = _backward
#         # self._prev = Set[Value]()

#         # for val in _children:
#         #     self._prev.add(val[])
#         self._op = _op


struct Value(KeyElement, Stringable):
    """Stores a single scalar value and its gradient."""
    
    var data: Float32
    var grad: Float32
    var _backward: fn () escaping -> None
    var _prev: Set[Value]
    var _op: String

    fn __init__(inout self, data: Float32, _backward: fn () escaping -> None, _children: List[Value] = List[Value](), _op: String = ""):
        self.data = data
        self.grad = Float32(0)

        # internal variables used for autograd graph constrution
        self._backward = _backward
        self._prev = Set[Value]()

        for val in _children:
            self._prev.add(val[])
        self._op = _op

    """
        Had to add the following for `KeyElement` for Set usage with Value
        __moveinit__
        __hash__
        __eq__
        __ne__

    """

    fn set_grad(inout self, val: Float32):
        self.grad = val

    fn __moveinit__(inout self: Self, owned existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._backward = existing._backward
        # For some reason the below is not working, this is recommended by kapa.ai
        # self._prev = existing._prev 

        self._prev = Set[Value]()

        for element in existing._prev:
            self._prev.add(element[])

        self._op = existing._op
    
    fn __hash__(self) -> Int:
        return 1

    fn __eq__(self, existing: Self) -> Bool: # SUS
        return True

    fn __ne__(self, existing: Self) -> Bool: # SUS
        return True
    
    fn __str__(self) -> String:
        return str(self.data)

    fn __copyinit__(inout self, existing: Self):
        self.data = existing.data
        self.grad = existing.grad
        self._backward = existing._backward
        self._prev = Set[Value]()

        for element in existing._prev:
            self._prev.add(element[])

        self._op = existing._op


    fn __add__(inout self, inout other: Value) -> Value:
        # print('asdfsasfd')
        var child = List[Value](self, other)
        # print('asdfsasfd')
        var out = Value(self.data + other.data, dummy_backward, child, '+')
        # print('asdfsasfd')

        """
            z = x + y
            l = 2*z

            dl/dz = 2
            dl/dx = dl/dz * dz/dx = 2 * 1 -> grad of z 
        """
        fn _backward() escaping -> None: # Hmm, gives error without raises
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    fn __add__(inout self, inout other: Float32) raises -> Value:
        var tmp = Value(other, dummy_backward)
        var child = List[Value](self, tmp)
        var out = Value(self.data + tmp.data, dummy_backward, child, '+')

        """
            z = x + y
            l = 2*z

            dl/dz = 2
            dl/dx = dl/dz * dz/dx = 2 * 1 -> grad of z 
        """
        fn _backward() escaping -> None: # Hmm, gives error without raises
            self.grad += out.grad
            tmp.grad += out.grad
        out._backward = _backward

        return out
    
    fn __mul__(inout self, inout other: Value) raises -> Value:
        var child = List[Value](self, other)
        var out = Value(self.data * other.data, dummy_backward, child, '*')

        """
            z = x * y
            l = 2*z

            dl/dz = 2
            dl/dx = dl/dz * dz/dx = 2 * y -> grad of z * other.data
        """
        fn _backward() -> None:
            self.grad += other.data * out.grad    
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    fn __mul__(inout self, inout other: Float32) raises -> Value:
        var tmp = Value(other, dummy_backward)
        var child = List[Value](self, tmp)
        var out = Value(self.data * tmp.data, dummy_backward, child, '*')

        """
            z = x * y
            l = 2*z

            dl/dz = 2
            dl/dx = dl/dz * dz/dx = 2 * y -> grad of z * other.data
        """
        fn _backward() -> None:
            self.grad += tmp.data * out.grad    
            tmp.grad += self.data * out.grad

            print(tmp.data, out.grad, self.grad, out.data)
        out._backward = _backward

        return out
    
    fn __pow__(inout self, other: Float32) raises -> Value:
        var child = List[Value](self,)
        # Only considering Int for simplicity 
        var out = Value(self.data ** other, dummy_backward, child, '**' + str(other))

        """
            z = x**y
            l = 2*z
            dl/dz = 2
            dl/dx = dl/dz * dz/dx = 2 * y*x**(y-1)
        """
        fn _backward() -> None:
            self.grad += (other * self.data ** (other - 1) * out.grad)
        
        out._backward = _backward
    
        return out

    fn relu(inout self) raises -> Value:
        var child = List[Value](self,)
        var out = Value(Float32(0) if self.data < 0 else self.data, dummy_backward, child, 'ReLU')

        """
            for relu
                x = y (for > 0) | therefore, 1
                y = 0 (for <= 0) | therefor, 0
        """
        fn _backward() -> None:
            print('This is relu backward!!', repr(self))
            self.grad += out.grad if out.data > 0 else 0
        
        out._backward = _backward

        return out

    # fn backward(inout self):
    #     # topological order all of the children in the graph

    #     var topo = []
    #     var visited = Set[Value]()

    #     fn build_topo(inout v: Value) -> None:
    #         if visited.__contains__(v):
    #             pass

    #     # fn build_topo(inout v: Value) -> None:
    #     #     if v not in visited:
    #     #         visited.add(v)
    #     #         for child in v._prev:
    #                 # build_topo(child[])

    # Hmm, type is not needed as it seems to infers from the other dunder method it is using
    # Nope, there was some otehr issue. Requires mentioning the return type
    fn __neg__(inout self: Self) raises -> Value: # -1 * self
        var neg = -Float32(1)
        return self * neg
        # return self * -Float32(1)
    
    fn __radd__(inout self, inout other: Value) raises -> Value: # other + self
        return self + other
    
    fn __sub(inout self, inout other: Value) raises -> Value: # self - other
        var tmp = -other
        return self + tmp
        
        # # the following doesn't work, issues in add
        # return self + (-other)

    fn __rsub__(inout self, inout other: Value) raises -> Value: # other - self
        var tmp = -self
        return other + tmp

    fn __rmul__(inout self, inout other: Value) raises -> Value: # other * self
        return self * other

    fn __rmul__(inout self, inout other: Float32) raises -> Value: # other * self
        return self * other
    
    fn __truediv(inout self, inout other: Value) raises -> Value: # self / other
        var pow = other**Float32(1)
        return self * (pow)
    
    fn __rtruediv__(inout self, inout other: Value) raises -> Value: # other / self
        var tmp = self**Float32(-1)
        return other * tmp
        # # Doesn't work
        # return other * self**-1
    
    fn __repr__(self) -> String:
        return "Value(data = " + str(self.data) + ", grad = " + str(self.grad) + ")"

fn dummy_backward() -> None:
    return None

fn main():
    var a = Float32(random.random_float64())
    var b = 1
    print(a)

    # alias fnc = fn () raises  -> None
    var val = Value(a, dummy_backward)
    # print(val.a)


        
