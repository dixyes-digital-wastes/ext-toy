--TEST--
Extra validation paths and oversized q8_0 shapes
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}

show(fn() => Toy\Tensor::fromBytes('f16', [1], "\0\0"));
show(fn() => Toy\Tensor::fromBytes('f32', [], ''));
$small = Toy\Tensor::zeros('f32', [1]);
show(fn() => $small->reshape([4294967295, 4294967295]));

$m = 65536;
$k = 32;
$weight = Toy\Tensor::full('q8_0', [$m, $k], 1.0);
$vector = Toy\Tensor::full('f32', [$k], 2.0);
Toy\Tensor\kernel_stats(true);
$product = Toy\Tensor\matvec_q8_0_f32($weight, $vector, $m, $k);
$floats = test_floats($product);
var_dump(abs($floats[0] - 64.0) < 0.02);
var_dump(abs($floats[$m - 1] - 64.0) < 0.02);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 0);
var_dump($stats['generation_failures'] >= 1);
var_dump($stats['scalar_calls'] === 1);

$m = 70;
$n = 33;
$k = 64;
$wv = [];
for ($i = 0; $i < $m * $k; $i++) {
    $wv[] = test_value($i, 7);
}
$xv = [];
for ($i = 0; $i < $k * $n; $i++) {
    $xv[] = test_value($i, 8);
}
$w = test_tensor([$m, $k], $wv)->cast('q8_0');
$x = test_tensor([$k, $n], $xv);
$out = test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
$blocks = test_q8_blocks($w->toBytes());
[$ref, $absSums] = test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
var_dump(test_q8_metrics($out, $ref, $absSums)['maxScaled'] < 4e-6);

echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
