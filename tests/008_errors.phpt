--TEST--
Error paths through the frameless handlers and the zif path
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
$x = tensor([2], [3.0, 4.0]);

show(fn() => Toy\Tensor\add($a, 'x'));
show(fn() => Toy\Tensor\add([], $b));
show(fn() => Toy\Tensor\scale($a, []));
show(fn() => Toy\Tensor\rmsnorm($x, tensor([2], [1.0, 2.0]), []));
show(fn() => Toy\Tensor\add($a, $flat));
show(fn() => Toy\Tensor\rmsnorm($x, tensor([1], [1.0])));
show(fn() => Toy\Tensor::zeros('q8_0', [32]));
show(fn() => Toy\Tensor\matmul_f32($a, $b, 3, 2, 2));
show(fn() => Toy\Tensor\kernel_isa('sse'));

$dynamic_add = 'Toy\\Tensor\\add';
$dynamic_scale = 'Toy\\Tensor\\scale';
show(fn() => $dynamic_add($a, 'x'));
show(fn() => $dynamic_scale($a, []));
show(fn() => $dynamic_add($a, $flat));
echo "done\n";
?>
--EXPECT--
TypeError
TypeError
TypeError
TypeError
ValueError
ValueError
ValueError
ValueError
ValueError
TypeError
TypeError
ValueError
done
