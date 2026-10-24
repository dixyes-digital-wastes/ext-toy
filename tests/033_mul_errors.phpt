--TEST--
mul and silu reject mismatched dtypes, shapes and out tensors
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

$a = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
$flat = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$q8 = tensor([2, 32], array_fill(0, 64, 1.0))->cast('q8_0');
$wide = tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

show(fn() => Toy\Tensor\mul($q8, $q8));
show(fn() => Toy\Tensor\mul($a, $q8));
show(fn() => Toy\Tensor\mul($a, $flat));
show(fn() => Toy\Tensor\mul($a, $wide));
show(fn() => Toy\Tensor\mul($a, $b, $q8));
show(fn() => Toy\Tensor\mul($a, $b, $wide));
show(fn() => Toy\Tensor\mul($a, $b, 1));
show(fn() => Toy\Tensor\mul(1, $b));
show(fn() => Toy\Tensor\mul($a));
show(fn() => Toy\Tensor\mul());

show(fn() => Toy\Tensor\silu($q8));
show(fn() => Toy\Tensor\silu($a, $q8));
show(fn() => Toy\Tensor\silu($a, $wide));
show(fn() => Toy\Tensor\silu($a, 1));
show(fn() => Toy\Tensor\silu(1));
show(fn() => Toy\Tensor\silu());

$before = $a->toBytes();
show(fn() => Toy\Tensor\mul($a, $b, $q8));
var_dump($a->toBytes() === $before);

Toy\Tensor\kernel_stats(true);
$in_place = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$reference = Toy\Tensor\silu($in_place);
var_dump(Toy\Tensor\silu($in_place, $in_place) === $in_place);
var_dump($in_place->toBytes() === $reference->toBytes());

$dynamic = 'Toy\\Tensor\\silu';
show(fn() => $dynamic($a, 1));
echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
TypeError
TypeError
ArgumentCountError
ArgumentCountError
ValueError
ValueError
ValueError
TypeError
TypeError
ArgumentCountError
ValueError
bool(true)
bool(true)
bool(true)
TypeError
done
