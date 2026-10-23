--TEST--
q8_0 kernels stay correct under the opcache optimizer
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

$isas = ['scalar'];
if (Toy\Tensor\kernel_stats()['avx2_available']) {
    $isas[] = 'avx2';
}

$ok = true;
foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    Toy\Tensor\kernel_stats(true);

    $m = 17;
    $n = 8;
    $k = 128;
    $wv = [];
    for ($i = 0; $i < $m * $k; $i++) {
        $wv[] = test_value($i, $m + $k);
    }
    $xv = [];
    for ($i = 0; $i < $k * $n; $i++) {
        $xv[] = test_value($i, $n + 1);
    }
    $w = test_tensor([$m, $k], $wv)->cast('q8_0');
    $x = test_tensor([$k, $n], $xv);
    $blocks = test_q8_blocks($w->toBytes());

    $got = test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
    [$ref, $absSums] = test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
    $metrics = test_q8_metrics($got, $ref, $absSums);
    $ok = $ok && $metrics['maxScaled'] < 4e-6;

    $stats = Toy\Tensor\kernel_stats();
    $ok = $ok && $stats['generated'] > 0;
}

var_dump($ok);
echo "done\n";
?>
--EXPECT--
bool(true)
done
