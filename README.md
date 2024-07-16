# MMgrad

Micrograd implementation in Mojo Programming Language

Resources which helped a lot:

- [micrograd](https://github.com/karpathy/micrograd)
- [mojograd](https://github.com/automata/mojograd)
- Also, kapa.ai (Modular Discord Bot)


## Included

- [x] Value
- [x] Neuron
- [x] Layer
- [x] MLP
- [x] Make moons dataset for mojo
- [x] Train fn
- [x] Plots
- [x] Add ipynb with Value, Neuron, Layer, MLP and backprop example

## TOC

| File | Contents |
| ------------- | ------------- |
| engine.mojo  | Value, operations and grad calculation for operations  |
| nn.mojo  | Neuron, Layer & MLP |
| play.mojo  | Dataset creation, Train, utils and main function for training  |
| test.ipynb  | Value, grad, Neuron, Layer & MLP testing  |
| attempts/*  | Mistakes and failed attempts  |