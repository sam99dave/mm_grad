from engine import Value
from nn import MLP
from python import Python, PythonObject

fn py_to_f32(x: PythonObject) -> Float32:
    var f64: Float64 = x.to_float64()
    return f64.cast[DType.float32]()

fn mojo_pls_make_moons(n_samples: Int = 100) raises -> (Pointer[Pointer[Pointer[Value]]], Pointer[Pointer[Value]]):
    var sklearn = Python.import_module("sklearn.datasets")
    var out = sklearn.make_moons(n_samples, noise = 0.1)

    var npy_X = out[0]
    var npy_y = out[1]

    # in testing for one sample it was Pointer[Pointer[Value]]
    var X = Pointer[Pointer[Pointer[Value]]].alloc(n_samples)
    for i in range(n_samples):
        var tmp_sample_ptr: Pointer[Pointer[Value]] = Pointer[Pointer[Value]].alloc(2)
        for j in range(2): # for X out points
            var tmp_x: Float32 = py_to_f32(npy_X[i][j])
            var tmp_val: Value = Value(tmp_x)
            var tmp_ptr: Pointer[Value] = Pointer[Value].alloc(1)
            tmp_ptr.store(tmp_val)
            tmp_sample_ptr.store(j, tmp_ptr)
        X.store(i, tmp_sample_ptr)

    var y: Pointer[Pointer[Value]] = Pointer[Pointer[Value]].alloc(n_samples)
    for i in range(n_samples):
        var tmp_ptr: Pointer[Value] = Pointer[Value].alloc(1)
        var tmp_y: Float32 = py_to_f32(npy_y[i])
        var tmp_val: Value = Value(tmp_y * 2 - 1)
        tmp_ptr.store(tmp_val)
        
        y.store(i, tmp_ptr)

    return X, y

fn train(inout model: MLP, inout X: Pointer[Pointer[Pointer[Value]]], inout y: Pointer[Pointer[Value]], inout n_samples: Int, inout epochs: Int):
    var const: Float32 = 1
    var alpha: Float32 = 1e-4
    for i in range(epochs):

        # Forward Pass
        var score_ptr: Pointer[Pointer[Value]] = Pointer[Pointer[Value]].alloc(n_samples)
        for idx in range(n_samples):
            var tmp_x = X.load(idx)
            # print(tmp_x.load(0).load().data.load())
            # print(tmp_x.load(1).load().data.load())
            # print("----")
            var out_ptr: Pointer[Pointer[Value]] = model.forward(tmp_x)

            # print("out score:", out_ptr.load().load().data.load())
            score_ptr.store(idx, out_ptr.load())
        
        var loss: Value = Value(0)
        for idx in range(n_samples):
            var tmp_score: Value = score_ptr.load(idx).load()
            var tmp_y: Value = y.load(idx).load()
            # print("Score: ", tmp_score.data.load())
            var tmp = -tmp_y
            tmp = tmp * tmp_score
            tmp = tmp + const
            # print("Before relu: ",tmp.data.load())
            tmp = tmp.relu()
            # print("After relu: ",tmp.data.load())
            loss = loss + tmp
        
        var len = Float32(n_samples)
        var data_loss: Value = loss / len

        var loss_p: Value = Value(0)
        for item in model.parameters():
            var tmp_val: Value = item[].load()
            tmp_val = tmp_val * tmp_val
            loss_p = loss_p + tmp_val
        
        var reg_loss: Value = loss_p * alpha

        var total_loss: Value = data_loss + reg_loss

        var accuracy: Float32 = 0

        for idx in range(n_samples):
            var tmp_score: Value = score_ptr.load(idx).load()
            var tmp_y: Value = y.load(idx).load()

            # print("tmp_score:", tmp_score.data.load())
            # print("tmp_y:", tmp_y.data.load())

            var check1: Bool = tmp_score.data.load() > 0
            var check2: Bool = tmp_y.data.load() > 0
            var check3: Bool = check1 == check2

            if check3:
                accuracy += 1
        
        accuracy = accuracy / n_samples

        print("Epochs: ",i," | loss - ",total_loss.data.load()," | acc - ",accuracy * 100)

        # Backward Pass
        model.zero_grad()
        total_loss.backward()

        var learning_rate: Float32 = 1.0 - 0.9*i/100

        # parameter update
        for item in model.parameters():
            var tmp_val: Value = item[].load() # this loads a copy of the value 
            # print('Grad: ', tmp_val.grad.load())
            var update: Float32 = tmp_val.data.load() - tmp_val.grad.load() * learning_rate
            item[].load().data.store(update)

        score_ptr.free()

    

fn main():

    try:
        var out = mojo_pls_make_moons(100)
        var X = out[0]
        var y = out[1]
        # print(X)
        # print(y)

        var nin: Int = 2
        var nouts: List[Int] = List[Int](16, 16, 1)
        var mlp: MLP = MLP(nin, nouts)

        var n_samples: Int = 100
        var epochs: Int = 100

        train(mlp, X, y, n_samples, epochs)
    
    except:
        print("In except")