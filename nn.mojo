from engine import Value
from memory.unsafe import Pointer
from random import random_float64 # float32 variant not available

# mojo structs doesn't support inheritance, trait can be used but it wont make much
# sense for implementing Module
# Implementing Neuron as Pointer to Pointers is better than List of Pointers,
# Pointer[Value] is possible as its `trivial` whereas Neuron won't be 
# List[Pointer[Value]] cannot be included as `trivial`.
# This will limit the Pointer[Neuron] for Layer 
# Therefore, instead of jumping between Pointer & List, keeping everything as Pointer

@register_passable('trivial')
struct Neuron:
    var w: Pointer[Pointer[Value]] # DONE:: This can also be a pointer that points to a pointer allocating nin memory locations, this works exactly like List 
    var b: Pointer[Value]
    var nonlin: Bool
    var nin: Int

    fn  __init__(inout self, nin: Int, nonlin: Bool = True):
        self.w = Pointer[Pointer[Value]].alloc(nin)

        for idx in range(nin):
            var ptr_v: Pointer[Value] = Pointer[Value].alloc(1)
            var val: Value = Value(random_float64(-1, 1).cast[DType.float32]()) # This returns type SIMD[float32, 1] ~ Float32
            ptr_v.store(val)
            self.w.store(idx, ptr_v)
            print('w is being prepped - ', idx) # DEBUG
        
        self.b = Pointer[Value].alloc(1)
        var val: Float32 = 0

        self.b.store(Value(val))

        self.nonlin = nonlin
        self.nin = nin
    
    # There is no __call__ in mojo 
    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Value]:
        # # length of self.w and x should be same
        # # The below check won't work as its no longer a List
        # # TODO:: Add this check back using Pointer
        # if self.nin != len(x):
        #     print('len of self.nin = ',self.nin,' is not same as len of x = ',len(x))
        #     return Pointer[Value]()
        
        var total: Value = Value(0)
        for i in range(self.nin):
            var tmp_w = self.w.load(i) # Return a Pointer
            var tmp_x = x.load(i) # Return a Pointer
            var val: Value = tmp_w.load()

            var tmp: Value = val * tmp_x

            total = total + tmp
        
        total = total + self.b

        var ptr_out: Pointer[Value] = Pointer[Value].alloc(1)

        if self.nonlin:
            ptr_out.store(total.relu())
        else:
            ptr_out.store(total)
        
        return ptr_out
    
    fn parameters(inout self) -> List[Pointer[Value]]:
        var all_ptr: List[Pointer[Value]] = List[Pointer[Value]]()

        for i in range(self.nin):
            all_ptr.append(self.w.load(i))
        
        all_ptr.append(self.b)

        return all_ptr
    
    ## Better to have it in MLP, where all the parameter are available for the network
    # fn zero_grad(inout self):
    #     for item in self.parameters():
    #         item[].load().grad.store(0)
        
    fn print(inout self):
        print("ReLU " if self.nonlin else "Linear ", "Neuron(",self.nin,")")


@register_passable('trivial')
struct Layer:
    var neurons: Pointer[Pointer[Neuron]]
    var nin: Int
    var nout: Int
    var nonlin: Bool

    fn __init__(inout self, nin: Int, nout: Int, nonlin: Bool = True):
        self.nonlin = nonlin
        self.nin = nin
        self.nout = nout
        self.neurons = Pointer[Pointer[Neuron]].alloc(self.nout)

        for idx in range(self.nout):
            var tmp_neu: Neuron = Neuron(self.nin, nonlin=self.nonlin)
            var tmp_ptr: Pointer[Neuron] = Pointer[Neuron].alloc(1)
            tmp_ptr.store(tmp_neu)

            self.neurons.store(idx, tmp_ptr)
        

    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Pointer[Value]]:
        var out: Pointer[Pointer[Value]] = Pointer[Pointer[Value]].alloc(self.nout)

        for idx in range(self.nout):
            var neuron: Neuron = self.neurons.load(idx).load()
            var out_ptr: Pointer[Value] = neuron.forward(x)
            out.store(idx, out_ptr)
        
        return out

    fn parameters(inout self) -> List[Pointer[Value]]:
        var all_ptr: List[Pointer[Value]] = List[Pointer[Value]]()

        for idx in range(self.nout):
            var tmp: Neuron = self.neurons.load(idx).load()
            all_ptr.extend(tmp.parameters())

        return all_ptr

    fn print(inout self):
        print("Layer | nin - ", self.nin, " | nout - ", self.nout, " | nonlin - ", self.nonlin, " | parameter len - ", len(self.parameters()))


struct MLP:
    var layers: Pointer[Pointer[Layer]]
    var nin: Int
    var nouts: List[Int]
    var sizes: List[Int]
    var n_layers: Int

    fn __init__(inout self, nin: Int, nouts: List[Int]):
        self.nin = nin # e.g 2
        self.nouts = nouts # e.g [16, 16, 1]
        self.n_layers = len(nouts)

        self.sizes = List[Int](self.nin)
        self.sizes.extend(self.nouts) # [2, 16, 16, 1] - using this will create layers

        self.layers = Pointer[Pointer[Layer]].alloc(self.n_layers)

        # Acc to above example
        # This will create 3 layers
        # 2x16 | 16x16 | 16x1
        # Non-Linearity will be skipped only to the last layer
        for idx in range(self.n_layers):
            var lay: Layer = Layer(
                self.sizes[idx],
                self.sizes[idx + 1],
                nonlin = idx != self.n_layers
            )

            var lay_ptr: Pointer[Layer] = Pointer[Layer].alloc(1)
            lay_ptr.store(lay)

            self.layers.store(idx, lay_ptr)
    

    fn forward(inout self, inout x: Pointer[Pointer[Value]]) -> Pointer[Pointer[Value]]:
        var out: Pointer[Pointer[Value]] = x
        for idx in range(self.n_layers):
            var lay: Layer = self.layers.load(idx).load()
            out = lay.forward(out) 
        
        return out
    
    fn parameters(inout self) -> List[Pointer[Value]]:
        var all_ptr: List[Pointer[Value]] = List[Pointer[Value]]()

        for idx in range(self.n_layers):
            var tmp: Layer = self.layers.load(idx).load()
            all_ptr.extend(tmp.parameters())

        return all_ptr

    fn print(inout self):
        # prepare nouts string
        var nouts_str: String = "["
        for item in self.nouts:
            nouts_str += str(item[]) 
            nouts_str += ","
        
        nouts_str += "]"

        print("MLP | nin - ", self.nin, " | nouts - ", nouts_str, " | parameter len - ", len(self.parameters()))
        
        
    fn zero_grad(inout self):
        for item in self.parameters():
            item[].load().grad.store(0.0)