--TEST--
Output reuse stays frameless and correct under the opcache optimizer
--EXTENSIONS--
toy
--SKIPIF--
<?php
if (!function_exists('opcache_get_status')) {
    die('skip opcache is not built in');
}
$status = @opcache_get_status(false);
if (!$status || empty($status['opcache_enabled'])) {
    die('skip opcache is not enabled for CLI');
}
?>
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

$a = tensor([4, 4], range(1, 16));
$b = tensor([4, 4], range(2, 17));

Toy\Tensor\kernel_stats(true);
$plain = Toy\Tensor\add($a, $b);
$out = Toy\Tensor::zeros('f32', [4, 4]);
$reuse = Toy\Tensor\add($a, $b, $out);
$scale_out = Toy\Tensor::zeros('f32', [4, 4]);
$scaled = Toy\Tensor\scale($a, 0.5, $scale_out);
$matmul_out = Toy\Tensor::zeros('f32', [4, 4]);
$matmul = Toy\Tensor\matmul_f32($a, $b, 4, 4, 4, $matmul_out);
$stats = Toy\Tensor\kernel_stats();

var_dump($reuse === $out);
var_dump($reuse->toBytes() === $plain->toBytes());
var_dump($scaled === $scale_out);
var_dump($matmul === $matmul_out);
var_dump($stats['flf_calls'] >= 3);
var_dump($stats['generated'] > 0);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
