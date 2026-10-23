--TEST--
Frameless calls stay frameless and correct under the opcache optimizer
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
$x = tensor([8], range(1, 8));
$w = tensor([8], range(1, 8));

Toy\Tensor\kernel_stats(true);
$frameless = Toy\Tensor\add($a, $b)->toBytes();
$dynamic = 'Toy\\Tensor\\add';
$zif = $dynamic($a, $b)->toBytes();
$stats = Toy\Tensor\kernel_stats();

var_dump($frameless === $zif);
var_dump($stats['flf_calls'] > 0);
var_dump($stats['zif_calls'] > 0);

$norm_kernel = Toy\Tensor\rmsnorm($x, $w)->toBytes();
Toy\Tensor\kernel_enable(false);
$norm_scalar = Toy\Tensor\rmsnorm($x, $w)->toBytes();
Toy\Tensor\kernel_enable(true);
var_dump($norm_kernel === $norm_scalar);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
done
