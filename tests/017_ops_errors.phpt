--TEST--
Operator argument, dtype and shape error paths
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
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

$a = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
$flat = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$vector = tensor([2], [3.0, 4.0]);
$q8 = tensor([2, 32], array_fill(0, 64, 1.0))->cast('q8_0');
$q8wrong = tensor([4, 32], array_fill(0, 128, 1.0))->cast('q8_0');
$x32 = tensor([32, 1], array_fill(0, 32, 1.0));

show(fn() => Toy\Tensor\add($a, $q8));
show(fn() => Toy\Tensor\add($a, $flat));
show(fn() => Toy\Tensor\add($a, tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
show(fn() => Toy\Tensor\add(1, $b));
show(fn() => Toy\Tensor\add($a));
show(function () use ($a, $b): void {
    $f = 'Toy\\Tensor\\add';
    $f($a, $b);
    echo "dynamic ok\n";
});
show(fn() => Toy\Tensor\scale($q8, 2.0));
show(fn() => Toy\Tensor\scale($a, []));
show(fn() => Toy\Tensor\rmsnorm($q8, $vector));
show(fn() => Toy\Tensor\rmsnorm($a, $vector));
show(fn() => Toy\Tensor\rmsnorm($vector, $a));
show(fn() => Toy\Tensor\rmsnorm($vector, tensor([3], [1.0, 1.0, 1.0])));
show(fn() => Toy\Tensor\rmsnorm($vector, $vector, []));

show(fn() => Toy\Tensor\matmul_f32($q8, $x32, 2, 1, 32));
show(fn() => Toy\Tensor\matmul_f32($flat, $b, 4, 2, 2));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 3, 2, 2));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 3, 2));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 0, 2, 2));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, -1));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2));
show(fn() => Toy\Tensor\matmul_f32(1, $b, 2, 2, 2));

show(fn() => Toy\Tensor\matvec_f32($flat, $vector, 4, 2));
show(fn() => Toy\Tensor\matvec_f32($a, $a, 2, 2));
show(fn() => Toy\Tensor\matvec_f32($a, $vector, 3, 2));
show(fn() => Toy\Tensor\matvec_f32($a, tensor([3], [1.0, 2.0, 3.0]), 2, 2));
show(fn() => Toy\Tensor\matvec_f32($a, $vector, 2, 0));

show(fn() => Toy\Tensor\matmul_q8_0_f32($a, $x32, 2, 1, 32));
show(fn() => Toy\Tensor\matmul_q8_0_f32($q8, $q8, 2, 1, 32));
show(fn() => Toy\Tensor\matmul_q8_0_f32($q8, $x32, 4, 1, 32));
show(fn() => Toy\Tensor\matmul_q8_0_f32($q8, $x32, 2, 1, 64));
show(fn() => Toy\Tensor\matmul_q8_0_f32($q8, $x32, 2, 2, 32));
show(fn() => Toy\Tensor\matmul_q8_0_f32($q8, $x32, 2, 0, 32));

show(fn() => Toy\Tensor\matvec_q8_0_f32($a, $x32, 2, 32));
show(fn() => Toy\Tensor\matvec_q8_0_f32($q8, $x32, 2, 32));
show(fn() => Toy\Tensor\matvec_q8_0_f32($q8wrong, $x32, 2, 32));
show(fn() => Toy\Tensor\matvec_q8_0_f32($q8, tensor([31], array_fill(0, 31, 1.0)), 2, 32));
show(fn() => Toy\Tensor\matvec_q8_0_f32($q8, tensor([32, 1], array_fill(0, 32, 1.0)), 2, 32));

show(fn() => Toy\Tensor\kernel_stats([]));
show(fn() => Toy\Tensor\kernel_enable([]));
show(fn() => Toy\Tensor\kernel_isa([]));
show(fn() => Toy\Tensor\kernel_isa('sse'));
show(fn() => Toy\Tensor\kernel_isa(''));
$isa = Toy\Tensor\kernel_isa();
var_dump(in_array($isa, ['scalar', 'avx2'], true));
var_dump(Toy\Tensor\kernel_enable(true));

echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
TypeError
ArgumentCountError
dynamic ok
no exception
ValueError
TypeError
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
ArgumentCountError
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
TypeError
TypeError
TypeError
ValueError
ValueError
bool(true)
bool(true)
done
