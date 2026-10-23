# toy

A PHP extension that provides the `Toy\Tensor` class and 8 tensor operators.

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

// add and scale
$sum = Toy\Tensor\add($a, $b);
$scaled = Toy\Tensor\scale($sum, 3.0);

// Optional out tensor: the result goes into $out
$out = Toy\Tensor::zeros('f32', [4]);
Toy\Tensor\add($a, $b, $out);

// q8_0 storage is optional. The last dimension must be a multiple of 32
$c = Toy\Tensor::full('f32', [32], 1.0);
$q = $c->cast('q8_0');
$back = $q->cast('f32');
```

## License

MIT. See [LICENSE](LICENSE).
