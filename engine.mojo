from memory.unsafe import Pointer

@register_passable("trivial")
struct Value:
    """Stores a single scalar value and its gradient."""
    
    var data: Pointer[Float32] # pointer to float32
    var grad: Pointer[Float32]

    # # Commenting this as Pointer[Int] == Pointer[Value] throws error
    # # Won't be able to perform visited check in the backward pass
    # var l: Pointer[Int]
    # var r: Pointer[Int]

    var l: Pointer[Value]
    var r: Pointer[Value]
    var _op: StringRef

    fn __init__(inout self, inp_data: Float32):
        self.data = Pointer[Float32].alloc(1)
        self.data.store(inp_data)

        self.grad = Pointer[Float32].alloc(1)
        self.grad.store(0.0)

        self.l = Pointer[Value]()
        self.r = Pointer[Value]()

        self._op = ""


    # --> ADD #
    @always_inline
    fn __add__(inout self, inout other: Pointer[Value]) -> Value:
        """Pointer + Pointer."""
        var new_val: Value = Value(
            self.data.load() + other.load().data.load()
        )

        var ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        
        # The bitcast method is used to create a new pointer, new_ptr, that points to the same memory location but treats the pointee as a Int.
        new_val.l = ptr_l.bitcast[Value]()
        new_val.r = other.bitcast[Value]()

        new_val._op = "+"

        return new_val

    @always_inline
    fn __add__(inout self, inout other: Float32) -> Value:
        """Pointer + Float32."""
        var other_v: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other_v)

        return self + ptr_v
    
    @always_inline
    fn __add__(inout self, inout other: Value) -> Value:
        """Pointer + Value."""
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)

        # this add will trigger the pointer __add__
        return self + ptr_v


    # --> MUL #
    @always_inline
    fn __mul__(inout self, inout other: Pointer[Value]) -> Value:
        """Pointer * Pointer."""
        var new_val: Value = Value(
            self.data.load() * other.load().data.load()
        )

        var ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        
        # The bitcast method is used to create a new pointer, new_ptr, that points to the same memory location but treats the pointee as a Int.
        new_val.l = ptr_l.bitcast[Value]()
        new_val.r = other.bitcast[Value]()

        new_val._op = "*"

        return new_val

    @always_inline
    fn __mul__(inout self, inout other: Float32) -> Value:
        """Pointer * Float32."""
        var other_v: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other_v)

        return self * ptr_v
    
    @always_inline
    fn __mul__(inout self, inout other: Value) -> Value:
        """Pointer * Value."""
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)

        # this add will trigger the pointer __add__
        return self * ptr_v


    # --> POW #
    @always_inline
    fn __pow__(inout self, inout other: Pointer[Value]) -> Value:
        """Pointer ** Pointer."""
        var new_val: Value = Value(
            self.data.load() ** other.load().data.load()
        )

        var ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        
        # The bitcast method is used to create a new pointer, new_ptr, that points to the same memory location but treats the pointee as a Int.
        new_val.l = ptr_l.bitcast[Value]()
        new_val.r = other.bitcast[Value]()

        new_val._op = "**"

        return new_val

    @always_inline
    fn __pow__(inout self, inout other: Float32) -> Value:
        """Pointer ** Float32."""
        var other_v: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other_v)

        return self ** ptr_v
    
    @always_inline
    fn __pow__(inout self, inout other: Value) -> Value:
        """Pointer ** Value."""
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)

        # this add will trigger the pointer __add__
        return self ** ptr_v


    # --> RELU #
    @always_inline
    fn relu(inout self) -> Value:
        """Perform RELU."""
        var new_val: Value = Value(
            0 if self.data.load() < 0 else self.data.load()
        )

        var ptr_l: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_l.store(self)
        
        # The bitcast method is used to create a new pointer, new_ptr, that points to the same memory location but treats the pointee as a Int.
        new_val.l = ptr_l.bitcast[Value]()
        # For ReLU, we dont need the right pointer, so it would be null
        new_val._op = "ReLU"

        return new_val


    # --> Neg #
    @always_inline
    fn __neg__(inout self) -> Value:
        """Self * -1."""
        var data: Float32 = -1

        return self * data


    # --> rADD #
    @always_inline
    fn __radd__(inout self, inout other: Float32) -> Value:
        """Other + Self."""
        return self + other

    @always_inline
    fn __radd__(inout self, inout other: Pointer[Value]) -> Value:
        """Other + Self."""
        return self + other
    

    # --> sub #
    @always_inline
    fn __sub__(inout self, inout other: Pointer[Value]) -> Value:
        """Self - Other."""
        
        var val: Value = other.load()
        var data: Value = -val

        return self + data

    @always_inline
    fn __sub__(inout self, inout other: Float32) -> Value:
        """Self - Other."""
        var data: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(data)

        return self - ptr_v

    @always_inline
    fn __sub__(inout self, inout other: Value) -> Value:
        """Self - Other."""
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)

        return self - ptr_v


    # --> rsub #
    @always_inline
    fn __rsub__(inout self, inout other: Pointer[Value]) -> Value:
        """Other - Self."""
        var val: Value = other.load()
        return val - self

    @always_inline
    fn __rsub__(inout self, inout other: Float32) -> Value:
        """Other - Self."""
        var val: Value = Value(other)
        return val - self


    # --> rmul #
    @always_inline
    fn __rmul__(inout self, inout other: Pointer[Value]) -> Value:
        """Other * Self."""
        return self * other

    @always_inline
    fn __rmul__(inout self, inout other: Float32) -> Value:
        """Other * Self."""
        return self * other


    # --> truediv #
    @always_inline
    fn __truediv__(inout self, inout other: Pointer[Value]) -> Value:
        """Self / Other."""
        var val: Value = other.load()
        var powie: Float32 = -1
        var data: Value = val**powie
        return self * data

    @always_inline
    fn __truediv__(inout self, inout other: Value) -> Value:
        """Self / Other."""
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(other)

        return self / ptr_v

    @always_inline
    fn __truediv__(inout self, inout other: Float32) -> Value:
        """Self / Other."""
        var data: Value = Value(other)
        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(data)
        
        return self / ptr_v


    # --> rtruediv #
    @always_inline
    fn __rtruediv__(inout self, inout other: Float32) -> Value:
        """Other / Self."""
        var val: Value = Value(other)
        return val / self

    @always_inline
    fn __rtruediv__(inout self, inout other: Pointer[Value]) -> Value:
        """Other / Self."""
        var val: Value = other.load()
        return val / self

    
    ## BACKWARD PASS ##
    # Moving this to another file might be better

    # --> add #
    @staticmethod
    fn backward_add(inout ptr_v: Pointer[Value]):
        """Don't need to check for null pointer cases as it wont be null.
            Still checking for safety.

            z = x + y
            l = 2*z

            dl/dx = dl/dz * dz/dx == 2 * 1
        """

        var val: Value = ptr_v.load()
        if val.l == Pointer[Value]():
            return 
        
        # For safety check, bicasting it (not required though)
        var left: Value = val.l.bitcast[Value]().load()

        left.grad.store(left.grad.load() + val.grad.load())

        if val.r == Pointer[Value]():
            return

        var right: Value = val.l.bitcast[Value]().load()

        right.grad.store(right.grad.load() + val.grad.load())
    
    # --> mul #
    @staticmethod
    fn backward_mul(inout ptr_v: Pointer[Value]):
        """Don't need to check for null pointer cases as it wont be null.
            Still checking for safety.

            z = x * y
            l = 2*z

            dl/dx = dl/dz * dz/dx == 2 * y
        """

        var val: Value = ptr_v.load()

        if val.l == Pointer[Value]() or val.r == Pointer[Value]():
            return
        
        var left: Value = val.l.bitcast[Value]().load()
        var right: Value = val.r.bitcast[Value]().load()

        left.grad.store(
            left.grad.load() + (val.grad.load() * right.data.load())
        )
        right.grad.store(
            right.grad.load() + (val.grad.load() * left.data.load())
        )
    
    # --> pow #
    @staticmethod
    fn backward_pow(inout ptr_v: Pointer[Value]):
        """Don't need to check for null pointer cases as it wont be null.
            Still checking for safety.

            z = x ** y
            l = 2*z

            dl/dx = dl/dz * dz/dx == 2 * yx^(y-1)

            Also, though we take y to be a pointer, y in this case will be treated 
            as a Float32. We won't update its grad.
        """

        var val: Value = ptr_v.load()

        if val.l == Pointer[Value]() or val.r == Pointer[Value]():
            return

        var left: Value = val.l.bitcast[Value]().load()
        var right: Value = val.r.bitcast[Value]().load()

        left.grad.store(
            left.grad.load() + (
                val.grad.load() * right.data.load() * (
                    left.data.load() ** (right.data.load() - 1)
                )
            )
        )

    # --> relu #
    @staticmethod
    fn backward_relu(inout ptr_v: Pointer[Value]):
        """Right child will be null over here.

            z = x if x > 0
            z = 0 if x < 0
            l = 2*z

            dl/dx = dl/dz * dz/dx == 2 * (1 if x > 0 else 0)

            Also, though we take y to be a pointer, y in this case will be treated 
            as a Float32. We won't update its grad.
        """
        var val: Value = ptr_v.load()

        if val.l == Pointer[Value]():
            return 
        
        var left: Value = val.l.bitcast[Value]().load()

        left.grad.store(
            left.grad.load() + (
                val.grad.load() if left.data.load() > 0 else 0
            )
        )

    @staticmethod
    fn _backward(inout ptr: Pointer[Value]):
        if ptr == Pointer[Value]():
            return
        
        var op: String = ptr.load()._op

        if op == '':
            return
        elif op == '*':
            Value.backward_mul(ptr)
        elif op == '+':
            Value.backward_add(ptr)
        elif op == '**':
            Value.backward_pow(ptr)
        elif op == 'ReLU':
            Value.backward_relu(ptr)
        else:
            print('op registered not supported - ', op)

    @staticmethod
    fn build_topo(
        inout ptr_v: Pointer[Value],
        inout visited: List[Pointer[Value]],
        inout topo: List[Pointer[Value]]
    ):
        if ptr_v == Pointer[Value]():
            return

        var is_visited: Bool = False

        for item in visited:
            if ptr_v == item[]:
                is_visited = True
                break
        
        if not is_visited:
            visited.append(ptr_v)

            if ptr_v.load().l != Pointer[Value]():
                # bitcast returns a new pointer pointing to the same location
                # so ptr_l == ptr_v.load().l -> will be True
                var ptr_l: Pointer[Value] = ptr_v.load().l.bitcast[Value]()
                Value.build_topo(ptr_l, visited, topo)
            
            if ptr_v.load().r != Pointer[Value]():
                var ptr_r: Pointer[Value] = ptr_v.load().r.bitcast[Value]()
                Value.build_topo(ptr_r, visited, topo)
            
            topo.append(ptr_v)


    fn backward(inout self):
        """List works but Set gives error.
        List before mojo 24.2 was referred to as DynamicVector.
        push_back() is now replaced with append().
        """

        var visited: List[Pointer[Value]] = List[Pointer[Value]]()
        var topo: List[Pointer[Value]] = List[Pointer[Value]]()

        var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
        ptr_v.store(self)

        Value.build_topo(ptr_v, visited, topo)

        self.grad.store(1.0)

        for item in reversed(topo):
            Value._backward(item[])
        
        visited.clear() # Hmm, visited should be deleted by the compiler, clearing it shoulbn't be necessary
        topo.clear()

        ptr_v.free() # As this is a unsafe Pointer, user has to manage the deletion!
        

    @always_inline
    fn print(inout self):
        print("<Value", "data: Pointer[Float32] -> ",self.data.load(), "grad: Pointer[Float32] -> ",self.grad.load(), "_op: StringRef -> ",self._op, ">")
    

    @always_inline
    fn __repr__(inout self):
        print("<Value", "data: Pointer[Float32] -> ",self.data.load(), "grad: Pointer[Float32] -> ",self.grad.load(), "_op: StringRef -> ",self._op, ">")

    # # TODO:: Not working, throwing error
    # @always_inline
    # fn __str__(self) -> String: # for print(Value) to work
    #     return "<Value" + "data: Pointer[Float32] -> " + str(self.data.load()) + "grad: Pointer[Float32] -> " + str(self.grad.load()) + "_op: StringRef -> " + self._op, ">"
        




    

