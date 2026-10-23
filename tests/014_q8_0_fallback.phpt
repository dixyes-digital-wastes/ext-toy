--TEST--
q8_0 operations fall back to the scalar reference when the arena is disabled
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=0
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

$cases = [
    [1, 1, 32],
    [3, 2, 96],
    [5, 7, 128],
    [8, 15, 1024],
    [17, 16, 64],
];

$ok = true;
Toy\Tensor\kernel_stats(true);
foreach ($cases as [$m, $n, $k]) {
    $wv = [];
    for ($i = 0; $i < $m * $k; $i++) {
        $wv[] = toy_test_value($i, $m * 5 + $k);
    }
    $xv = [];
    for ($i = 0; $i < $k * $n; $i++) {
        $xv[] = toy_test_value($i, $n * 7 + 1);
    }
    $w = toy_test_tensor([$m, $k], $wv)->cast('q8_0');
    $x = toy_test_tensor([$k, $n], $xv);
    $blocks = toy_test_q8_blocks($w->toBytes());

    $got = toy_test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
    [$ref, $absSums] = toy_test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
    $metrics = toy_test_q8_metrics($got, $ref, $absSums);
    $ok = $ok && $metrics['maxScaled'] < 4e-6;

    $vector = array_slice($xv, 0, $k);
    $x1 = toy_test_tensor([$k], $vector);
    $got = toy_test_floats(Toy\Tensor\matvec_q8_0_f32($w, $x1, $m, $k));
    [$ref, $absSums] = toy_test_q8_matvec_ref($blocks, $vector, $m, $k);
    $metrics = toy_test_q8_metrics($got, $ref, $absSums);
    $ok = $ok && $metrics['maxScaled'] < 4e-6;
}

$stats = Toy\Tensor\kernel_stats();
var_dump($ok);
var_dump($stats['generated'] === 0);
var_dump($stats['kernel_calls'] === 0);
var_dump($stats['scalar_calls'] >= 10);

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena') && str_contains($info, 'disabled'));
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
