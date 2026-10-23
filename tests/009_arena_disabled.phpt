--TEST--
Scalar fallback when the kernel arena is disabled
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=0
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

$a = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
$x = tensor([2], [3.0, 4.0]);
$w = tensor([2], [1.0, 2.0]);

Toy\Tensor\kernel_stats(true);
$add = Toy\Tensor\add($a, $b);
$scale = Toy\Tensor\scale($a, 2.0);
$matmul = Toy\Tensor\matmul_f32($a, $b, 2, 2, 2);
$matvec = Toy\Tensor\matvec_f32($a, $x, 2, 2);
$norm = Toy\Tensor\rmsnorm($x, $w);

echo json_encode(array_values(unpack('g*', $add->toBytes()))), "\n";
echo json_encode(array_values(unpack('g*', $scale->toBytes()))), "\n";
echo json_encode(array_values(unpack('g*', $matmul->toBytes()))), "\n";
echo json_encode(array_values(unpack('g*', $matvec->toBytes()))), "\n";

$norm_values = array_values(unpack('g*', $norm->toBytes()));
$norm_scale = 1.0 / sqrt(25.0 / 2.0 + 1e-6);
var_dump(abs($norm_values[0] - 3.0 * $norm_scale) < 1e-6);
var_dump(abs($norm_values[1] - 8.0 * $norm_scale) < 1e-6);

$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 0);
var_dump($stats['kernel_calls'] === 0);
var_dump($stats['scalar_calls'] >= 5);

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena') && str_contains($info, 'disabled'));
echo "done\n";
?>
--EXPECT--
[6,8,10,12]
[2,4,6,8]
[19,22,43,50]
[11,25]
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
