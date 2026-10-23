# Examples

This doc explains how to download MNIST, train the example MLP on it, and read the results. Run everything from the repo root after [building](../README.md#build).

- [MLP on MNIST](#mlp-on-mnist)
  - [Getting the data](#getting-the-data)
  - [Usage](#usage)
  - [Reading the output](#reading-the-output)
  - [How it's built](#how-its-built)

## MLP on MNIST

`examples/mlp_mnist.cpp` trains an MLP on MNIST using only cuDL kernels, then reports test accuracy and time per step.

### Getting the data

```text
scripts/get_mnist.sh [DIR]
```

```sh
scripts/get_mnist.sh      # downloads to data/mnist
```

**Note:** On the cluster, run this on the login node. Compute nodes might not have internet.

### Usage

```text
build/examples/mlp_mnist [--data DIR] [--epochs N] [--batch N] [--hidden N,N,...]
                         [--optim adam|sgd] [--lr X] [--momentum X] [--seed N] [--target X]
```

```sh
build/examples/mlp_mnist
build/examples/mlp_mnist --optim sgd --epochs 10
build/examples/mlp_mnist --hidden 256           # one hidden layer
```

Flags:

```sh
--data data/mnist         # where get_mnist.sh put it
--epochs 5 --batch 128    # defaults
--hidden 512,256          # hidden layer sizes, so 784 -> 512 -> 256 -> 10
--optim adam --lr 1e-3    # sgd defaults to lr 0.05
--momentum 0.9            # sgd only
--seed 0                  # init + shuffle order
--target 0.97             # exit 1 if final test accuracy is below this
```

**Note:** The defaults are meant to clear the 97% target, so a plain run is a quick check that nothing is broken. `--target` makes it usable as a smoke test in a script or a Slurm job (`sbatch scripts/slurm/train.sbatch`).

### Reading the output

The program prints one line per epoch with the loss, test accuracy and ms/step, then the final accuracy and mean ms/step.

- Accuracy below the target usually means a kernel is wrong, not that training needs tuning. Run the [tests](../docs/testing.md) first.
- ms/step covers the whole step, including copying the batch to the GPU. To see where the time goes, [profile it with nsys](../docs/profiling.md#profiling-training-nsys).

### How it's built

The API looks like PyTorch, minus autograd, so backward is called on the model explicitly:

```cpp
nn::Rng rng(0);
nn::Sequential model;
model.add<nn::Linear>(784, 256, rng);
model.add<nn::ReLU>();
model.add<nn::Linear>(256, 10, rng);

nn::CrossEntropyLoss loss_fn;
optim::Adam opt(model.parameters(), 1e-3f);

Tensor loss = loss_fn.forward(model.forward(x), labels);
model.backward(loss_fn.backward());
opt.step();
```

`model.backward` runs the layers in reverse to compute the gradients, and `opt.step()` applies them.
