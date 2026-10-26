--TEST--
The activation, l2norm and conv1d argument, dtype, range and overlap errors
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
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
$q8 = tensor([1, 32], array_fill(0, 32, 1.0))->cast('q8_0');

/* Unary math: f32 only, out must match */
show(fn() => Toy\Tensor\sigmoid($q8));
show(fn() => Toy\Tensor\sigmoid());
show(fn() => Toy\Tensor\sigmoid($flat, tensor([5], range(1, 5))));
show(fn() => Toy\Tensor\sigmoid($flat, $q8));
show(fn() => Toy\Tensor\sigmoid($flat, 1));
show(fn() => Toy\Tensor\softplus($q8));
show(fn() => Toy\Tensor\softplus($flat, tensor([2, 3], range(1, 6))));

/* l2norm_rows needs a 2-D f32 input and a row range inside it */
show(fn() => Toy\Tensor\l2norm_rows($flat));
show(fn() => Toy\Tensor\l2norm_rows($q8->reshape([1, 32])));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, -1));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, 3));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, 1, -1));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, 1, 2));
show(fn() => Toy\Tensor\l2norm_rows($pair, [], 0));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, 0, null, tensor([3, 2], range(1, 6))));
show(fn() => Toy\Tensor\l2norm_rows($pair, 1e-6, 0, null, $q8));

/* conv1d_silu checks the two shapes against rows and kernel */
$w = tensor([4, 3], range(1, 12));
$x = tensor([3, 4], range(1, 12));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 0, null, tensor([3], [1.0, 2.0, 3.0])));
show(fn() => Toy\Tensor\conv1d_silu(tensor([3, 3], range(1, 9)), $x, 4, 3));
show(fn() => Toy\Tensor\conv1d_silu($w, tensor([4, 3], range(1, 12)), 4, 3));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 0, 3));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 0));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, -1));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 5));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 2, 3));
show(fn() => Toy\Tensor\conv1d_silu($q8, $x, 4, 3));
show(fn() => Toy\Tensor\conv1d_silu($w, $q8, 4, 3));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 0, null, $w));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 0, null, tensor([5], [1.0, 2.0, 3.0, 4.0, 5.0])));
show(fn() => Toy\Tensor\conv1d_silu($w, $x, 4, 3, 0, null, 1));

/* An empty slice is allowed and leaves the result untouched */
$empty = Toy\Tensor\l2norm_rows($pair, 1e-6, 2, 0);
var_dump(array_values(unpack('g*', $empty->toBytes())) === array_fill(0, 6, 0.0));
$empty = Toy\Tensor\conv1d_silu($w, $x, 4, 3, 4, 0);
var_dump(array_values(unpack('g*', $empty->toBytes())) === array_fill(0, 4, 0.0));

$result = Toy\Tensor\sigmoid($flat);
var_dump($result->shape() === [6]);
var_dump(Toy\Tensor\softplus($flat)->shape() === [6]);
echo "done\n";
?>
--EXPECT--
ValueError
ArgumentCountError
ValueError
ValueError
TypeError
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
TypeError
bool(true)
bool(true)
bool(true)
bool(true)
done
