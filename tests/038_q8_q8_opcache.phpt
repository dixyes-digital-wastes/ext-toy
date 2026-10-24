--TEST--
q8_0 x q8_0 kernels stay correct under the opcache optimizer
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
require __DIR__ . '/helpers/q8.php';

$dw = 0x2C00;
$dx = 0x3000;
$m = 9;
$n = 32;
$k = 64;
$blocks = intdiv($k, 32);

$wblocks = [];
for ($i = 0; $i < $m * $blocks; $i++) {
    $wblocks[] = ['d' => $dw, 'q' => test_q8_small(32, 7, $i + 6)];
}
$columns = [];
for ($j = 0; $j < $n; $j++) {
    $column = [];
    for ($b = 0; $b < $blocks; $b++) {
        $column[] = ['d' => $dx, 'q' => test_q8_small(32, 7, $j * $blocks + $b + 2)];
    }
    $columns[$j] = $column;
}
$w = Toy\Tensor::fromBytes('q8_0', [$m, $k], test_q8_pack($wblocks));
$x = Toy\Tensor::fromBytes('q8_0', [$k, $n], test_q8_pack(array_merge(...$columns)));
$vector = Toy\Tensor::fromBytes('q8_0', [$k], test_q8_pack($columns[3]));

$ok = true;
$rows = array_values(unpack('g*', Toy\Tensor\matvec_q8_0_q8_0($w, $vector, $m, $k)->toBytes()));
for ($i = 0; $i < $m; $i++) {
    $ok = $ok && $rows[$i] === test_q8_q8_dot($wblocks, $i, $columns[3], $blocks);
}

$values = array_values(unpack('g*', Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k)->toBytes()));
for ($i = 0; $i < $m; $i++) {
    for ($j = 0; $j < $n; $j++) {
        $ok = $ok && $values[$i * $n + $j] === test_q8_q8_dot($wblocks, $i, $columns[$j], $blocks);
    }
}

Toy\Tensor\kernel_enable(false);
$scalar = Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k);
Toy\Tensor\kernel_enable(true);
$ok = $ok && $scalar->toBytes() === Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k)->toBytes();

$ok = $ok && Toy\Tensor\kernel_stats()['kernel_calls'] > 0;
var_dump($ok);
echo "done\n";
?>
--EXPECT--
bool(true)
done
