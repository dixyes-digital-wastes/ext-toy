--TEST--
The rmsnorm_rows and shift_rows argument, dtype, range and shape errors
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function values(Toy\Tensor $tensor): array {
    return array_values(unpack('g*', $tensor->toBytes()));
}
function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}

$flat = tensor([6], range(1, 6));
$pair = tensor([2, 3], range(1, 6));
$weight = tensor([3], [1.0, 2.0, 3.0]);
$q8 = tensor([1, 32], array_fill(0, 32, 1.0))->cast('q8_0');

/* rmsnorm_rows takes a 2-D f32 input, a weight row of the same width and a
   row range inside the input */
show(fn() => Toy\Tensor\rmsnorm_rows($flat, $weight));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $flat));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, tensor([4], [1.0, 2.0, 3.0, 4.0])));
show(fn() => Toy\Tensor\rmsnorm_rows($q8, $weight));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $q8));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, -1));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, 3));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, 1, 2));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, [], 0));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, 0, null,
    tensor([3, 2], range(1, 6))));
show(fn() => Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, 0, null, 1));

/* shift_rows needs a 2-D state and a row of the same width */
show(fn() => Toy\Tensor\shift_rows($flat));
show(fn() => Toy\Tensor\shift_rows($q8));
show(fn() => Toy\Tensor\shift_rows($pair, tensor([2], [1.0, 2.0])));
show(fn() => Toy\Tensor\shift_rows($pair, tensor([3, 1], range(1, 4))));
show(fn() => Toy\Tensor\shift_rows($pair, $q8));
show(fn() => Toy\Tensor\shift_rows($pair, $weight, $flat));
show(fn() => Toy\Tensor\shift_rows($pair, $weight, tensor([3, 2], range(1, 6))));
show(fn() => Toy\Tensor\shift_rows($pair, 1));

/* An empty row range and a plain shift stay quiet */
$empty = Toy\Tensor\rmsnorm_rows($pair, $weight, 1e-6, 2, 0);
var_dump(values($empty) === array_fill(0, 6, 0.0));
$shifted = Toy\Tensor\shift_rows($pair, $weight);
var_dump(values($shifted) === [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
var_dump(values($pair) === [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
var_dump(Toy\Tensor\rmsnorm_rows($pair, $weight)->shape() === [2, 3]);
var_dump(Toy\Tensor\shift_rows($pair)->shape() === [2, 3]);
echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
TypeError
ValueError
TypeError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
TypeError
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
