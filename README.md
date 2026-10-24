# toy

A PHP extension that provides the `Toy\Tensor` class and 12 tensor operators.

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

// q8_0 storage is optional. The last dimension must be a multiple of 32
$c = Toy\Tensor::full('f32', [32], 1.0);
$q = $c->cast('q8_0');
$back = $q->cast('f32');

// q8_0 weights against q8_0 activations. The 32 value blocks, the f16 scales
// and the lane order follow llama.cpp
$weights = Toy\Tensor::full('f32', [4, 64], 1.0)->cast('q8_0');
$vector = Toy\Tensor::full('f32', [64], 2.0)->cast('q8_0');
$matvec = Toy\Tensor\matvec_q8_0_q8_0($weights, $vector, 4, 64);

// An activation matrix keeps one q8_0 vector of k values per column, so a
// [n, k] f32 input of rows becomes a [k, n] q8_0 tensor after the cast
$columns = Toy\Tensor::full('f32', [32, 64], 2.0)->cast('q8_0')->reshape([64, 32]);
$matmul = Toy\Tensor\matmul_q8_0_q8_0($weights, $columns, 4, 32, 64);
```

## License

MIT. See [LICENSE](LICENSE).
