# toy

A PHP extension that provides the `Toy\Tensor` class and 20 tensor operators.

## Requirements

- PHP 8.4 or later, with the development headers
- A compiler that supports C23

## Build

```bash
phpize
./configure --enable-toy
make -j
```

The build writes `modules/toy.so`.

## Test

```bash
make test TESTS=tests
```

## Usage

Load the extension:

```bash
php -n -d extension=$PWD/modules/toy.so -m
```

The output of `php -m` must contain `toy`.

```php
<?php

// Construct a tensor
$a = Toy\Tensor::full('f32', [4], 1.0);
$b = Toy\Tensor::full('f32', [4], 2.0);

// add, mul, scale and silu
$sum = Toy\Tensor\add($a, $b);
$product = Toy\Tensor\mul($a, $b);
$scaled = Toy\Tensor\scale($sum, 3.0);
$activated = Toy\Tensor\silu($product);

// Optional out tensor: the result goes into $out
$out = Toy\Tensor::zeros('f32', [4]);
Toy\Tensor\add($a, $b, $out);

// sigmoid and softplus are element-wise. Above 20 softplus is the identity
$squashed = Toy\Tensor\sigmoid($product);
$smooth = Toy\Tensor\softplus($product);

// The l2 norm of every row, with eps inside the square root. A row range selects a slice of rows, which is one head per row for the attention norm
$rows = Toy\Tensor::fromBytes('f32', [2, 4], pack('g*', 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0));
$normed = Toy\Tensor\l2norm_rows($rows);
$head = Toy\Tensor\l2norm_rows($rows, 1e-6, 1, 1);

// Every row against one shared weight row, which is the per-head norm of the demo, and again a row range selects the heads
$headWeight = Toy\Tensor::full('f32', [4], 2.0);
$headNorm = Toy\Tensor\rmsnorm_rows($rows, $headWeight);
$oneHeadNorm = Toy\Tensor\rmsnorm_rows($rows, $headWeight, 1e-6, 1, 1);

// A depthwise convolution over channels with SiLU on the result.
// The weights hold one kernel per channel, the input holds one row per tap and the last row of the input is the current step.
// The row range selects channels
$convWeights = Toy\Tensor::full('f32', [2, 4], 0.5);
$convInput = Toy\Tensor::full('f32', [4, 2], 1.0);
$conv = Toy\Tensor\conv1d_silu($convWeights, $convInput, 2, 4);

// shift_rows moves the rows of the convolution state up by one and writes the optional row into the last one.
// In place here, which is the state update of one step, and the next call reads the shifted state
$state = Toy\Tensor::full('f32', [4, 2], 0.0);
$step = Toy\Tensor::full('f32', [2], 1.0);
Toy\Tensor\shift_rows($state, $step, $state);
$pushed = Toy\Tensor\conv1d_silu($convWeights, $state, 2, 4);

// q8_0 storage is optional. The last dimension must be a multiple of 32
$c = Toy\Tensor::full('f32', [32], 1.0);
$q = $c->cast('q8_0');
$back = $q->cast('f32');

// Activation quantization returns the same blocks cast does, and rowBegin and rowEnd select a row range without copying it first.
// The quantization rule is the ggml one: d = amax / 127 and q = roundf(x / d)
$x = Toy\Tensor::full('f32', [8, 64], 2.0);
$activation = Toy\Tensor\quantize_q8_0($x);
$rows = Toy\Tensor\quantize_q8_0($x, 4, 8);

// q8_0 weights against q8_0 activations. The 32 value blocks, the f16 scales and the lane order follow llama.cpp
$weights = Toy\Tensor::full('f32', [4, 64], 1.0)->cast('q8_0');
$vector = Toy\Tensor::full('f32', [64], 2.0)->cast('q8_0');
$matvec = Toy\Tensor\matvec_q8_0_q8_0($weights, $vector, 4, 64);

// An activation matrix keeps one q8_0 vector of k values per column, so a [n, k] f32 input of rows becomes a [k, n] q8_0 tensor after the cast
$columns = Toy\Tensor::full('f32', [32, 64], 2.0)->cast('q8_0')->reshape([64, 32]);
$matmul = Toy\Tensor\matmul_q8_0_q8_0($weights, $columns, 4, 32, 64);

// The gated delta net recurrence.
// One state row is one value column and holds the keys, so the state is [heads, value_dim, key_dim] and a row is updated in place.
// The last four arguments select the local heads and value columns
$heads = 2;
$keyDim = 4;
$valueDim = 3;
$q = Toy\Tensor::full('f32', [$heads, $keyDim], 0.5);
$k = Toy\Tensor::full('f32', [$heads, $keyDim], 0.25);
$v = Toy\Tensor::full('f32', [$heads, $valueDim], 1.0);
$decay = Toy\Tensor::full('f32', [$heads], 0.5);
$beta = Toy\Tensor::full('f32', [$heads], 0.75);
$state = Toy\Tensor::zeros('f32', [$heads, $valueDim, $keyDim]);
$out = Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state,
    0, $heads, 0, $valueDim, 1.0 / sqrt($keyDim));
```

## License

MIT. See [LICENSE](LICENSE).
