--TEST--
Invalid out tensors and overlapping outputs are rejected
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
$x = tensor([2], [1.0, 2.0]);
$q8 = tensor([2, 32], array_fill(0, 64, 1.0))->cast('q8_0');

show(fn() => Toy\Tensor\add($a, $b, $q8));
show(fn() => Toy\Tensor\add($a, $b, tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
show(fn() => Toy\Tensor\add($a, $b, tensor([2, 2, 1], [1.0, 2.0, 3.0, 4.0])));
show(fn() => Toy\Tensor\scale($a, 2.0, $q8));
show(fn() => Toy\Tensor\scale($a, 2.0, tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));

show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, $q8));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, $a));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, $b));
show(fn() => Toy\Tensor\matvec_f32($a, $x, 2, 2, $x));
show(fn() => Toy\Tensor\matvec_f32($a, $x, 2, 2, tensor([3], [1.0, 2.0, 3.0])));

$qw = tensor([32, 32], array_fill(0, 1024, 1.0))->cast('q8_0');
$qx = tensor([32, 1], array_fill(0, 32, 1.0));
$qv = tensor([32], array_fill(0, 32, 1.0));
show(fn() => Toy\Tensor\matmul_q8_0_f32($qw, $qx, 32, 1, 32, $qx));
show(fn() => Toy\Tensor\matmul_q8_0_f32($qw, $qx, 32, 1, 32, $qw));
show(fn() => Toy\Tensor\matvec_q8_0_f32($qw, $qv, 32, 32, $qv));

$norm_x = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$norm_w = tensor([4], [1.0, 0.5, 2.0, 1.5]);
show(fn() => Toy\Tensor\rmsnorm_into($norm_x, $norm_w, $norm_w));
show(fn() => Toy\Tensor\rmsnorm_into($norm_x, $norm_w, $q8));
show(fn() => Toy\Tensor\rmsnorm_into($norm_x, $norm_w, tensor([3], [1.0, 2.0, 3.0])));

show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, 1));
show(fn() => Toy\Tensor\matvec_f32($a, $x, 2, 2, 1));
$dynamic = 'Toy\\Tensor\\add';
show(fn() => $dynamic($a, $b, 1));

$a_before = $a->toBytes();
$b_before = $b->toBytes();
show(fn() => Toy\Tensor\matmul_f32($a, $b, 2, 2, 2, $a));
var_dump($a->toBytes() === $a_before);
var_dump($b->toBytes() === $b_before);
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
bool(true)
bool(true)
done
