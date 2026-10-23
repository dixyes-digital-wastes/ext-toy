--TEST--
Shape, dtype, byte and reshape validation paths
--EXTENSIONS--
toy
--FILE--
<?php
function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}

show(fn() => Toy\Tensor::zeros('f16', [1]));
show(fn() => Toy\Tensor::zeros('bf16', [1]));
show(fn() => Toy\Tensor::zeros('i8', [1]));
show(fn() => Toy\Tensor::zeros('q8', [32]));
show(fn() => Toy\Tensor::zeros([], [1]));
show(fn() => Toy\Tensor::zeros('f32', 'shape'));
show(fn() => Toy\Tensor::zeros('f32', [1, 2, 3, 4, 5]));
show(fn() => Toy\Tensor::zeros('f32', []));
show(fn() => Toy\Tensor::zeros('f32', [0]));
show(fn() => Toy\Tensor::zeros('f32', [-1]));
show(fn() => Toy\Tensor::zeros('f32', [1.5]));
show(fn() => Toy\Tensor::zeros('f32', [true]));
show(fn() => Toy\Tensor::zeros('f32', ['a' => 1]));
show(fn() => Toy\Tensor::zeros('f32', [1 => 1, 0 => 2]));
show(fn() => Toy\Tensor::zeros('f32', [4294967296]));
show(fn() => Toy\Tensor::zeros('f32', [4294967295, 4294967295]));

show(fn() => Toy\Tensor::fromBytes('q8_0', [33], str_repeat("\0", 34)));
show(fn() => Toy\Tensor::fromBytes('f32', [4294967295, 4294967295], ''));
show(fn() => Toy\Tensor::fromBytes('f32', [2, 2], str_repeat("\0", 15)));

$matrix = Toy\Tensor::zeros('f32', [2, 3]);
show(fn() => $matrix->reshape([7]));
show(fn() => $matrix->reshape([]));
show(fn() => $matrix->reshape([1, 2, 3, 4, 5]));
show(fn() => $matrix->reshape([0, 6]));
show(fn() => $matrix->reshape([1.5, 4]));
show(fn() => $matrix->reshape(['x']));
show(fn() => $matrix->reshape([1 => 6, 0 => 1]));

$four = Toy\Tensor::zeros('f32', [1, 2, 3, 4]);
var_dump($four->shape());
var_dump($four->ndim());
var_dump($four->bytes());
var_dump(strlen($four->toBytes()));

$q8 = Toy\Tensor::zeros('q8_0', [2, 32]);
show(fn() => $q8->reshape([1, 64])->reshape([64, 1]));
var_dump($q8->shape());
var_dump($q8->bytes());
show(fn() => $q8->reshape([32, 2]));

$copy = $matrix->cast('f32');
var_dump($copy !== $matrix);
var_dump($copy->toBytes() === $matrix->toBytes());
show(fn() => $matrix->cast([]));

$wide = Toy\Tensor::zeros('f32', [1, 2, 3, 4]);
var_dump($wide->reshape([4, 6])->shape());

echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
ValueError
TypeError
TypeError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
array(4) {
  [0]=>
  int(1)
  [1]=>
  int(2)
  [2]=>
  int(3)
  [3]=>
  int(4)
}
int(4)
int(96)
int(96)
ValueError
array(2) {
  [0]=>
  int(1)
  [1]=>
  int(64)
}
int(68)
ValueError
bool(true)
bool(true)
TypeError
array(2) {
  [0]=>
  int(4)
  [1]=>
  int(6)
}
done
