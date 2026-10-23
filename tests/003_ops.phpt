--TEST--
Toy\Tensor scalar reference operations
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function show_floats(Toy\Tensor $tensor): void {
    $values = array_values(unpack('g*', $tensor->toBytes()));
    echo json_encode(array_map(fn(float $v): float => round($v, 6), $values)), "\n";
}
function bool_text(bool $value): string {
    return $value ? 'true' : 'false';
}

$a = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);

$matmul = Toy\Tensor\matmul_f32($a, $b, 2, 2, 2);
echo $matmul->dtype, ' ', json_encode($matmul->shape()), "\n";
show_floats($matmul);

$matvec = Toy\Tensor\matvec_f32($a, tensor([2], [5.0, 6.0]), 2, 2);
echo json_encode($matvec->shape()), "\n";
show_floats($matvec);

show_floats(Toy\Tensor\add($a, $b));
show_floats(Toy\Tensor\scale($a, 2.5));

$norm = Toy\Tensor\rmsnorm(tensor([2], [3.0, 4.0]), tensor([2], [1.0, 2.0]));
$norm_values = array_values(unpack('g*', $norm->toBytes()));
$norm_scale = 1.0 / sqrt(25.0 / 2.0 + 1e-6);
echo $norm->dtype, ' ', json_encode($norm->shape()), ' ',
    bool_text(abs($norm_values[0] - 3.0 * $norm_scale) < 1e-6), ' ',
    bool_text(abs($norm_values[1] - 8.0 * $norm_scale) < 1e-6), "\n";

$norm_zero = Toy\Tensor\rmsnorm(tensor([2], [3.0, 4.0]), tensor([2], [1.0, 2.0]), 0.0);
$zero_values = array_values(unpack('g*', $norm_zero->toBytes()));
echo bool_text(abs($zero_values[0] - 3.0 / sqrt(12.5)) < 1e-6), ' ',
    bool_text(abs($zero_values[1] - 8.0 / sqrt(12.5)) < 1e-6), "\n";

try { Toy\Tensor\matmul_f32($a, $b, 3, 2, 2); } catch (Throwable $e) { echo get_class($e), "\n"; }
try { Toy\Tensor\add($a, tensor([4], [1.0, 2.0, 3.0, 4.0])); } catch (Throwable $e) { echo get_class($e), "\n"; }
try { Toy\Tensor\rmsnorm(tensor([2], [3.0, 4.0]), tensor([1], [1.0])); } catch (Throwable $e) { echo get_class($e), "\n"; }
echo "done\n";
?>
--EXPECT--
f32 [2,2]
[19,22,43,50]
[2]
[17,39]
[6,8,10,12]
[2.5,5,7.5,10]
f32 [2] true true
true true
ValueError
ValueError
ValueError
done
