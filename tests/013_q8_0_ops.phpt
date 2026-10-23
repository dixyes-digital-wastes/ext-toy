--TEST--
q8_0 matmul and matvec match the dequantized double reference on both ISAs
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

$ms = [1, 2, 3, 5, 7, 8, 15, 16, 17];
$ns = [1, 2, 3, 5, 7, 8, 15, 16, 17];
$ks = [32, 64, 96, 128, 1024];

$isas = ['scalar'];
if (Toy\Tensor\kernel_stats()['avx2_available']) {
    $isas[] = 'avx2';
}

$ok = true;
$usedKernels = true;
foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    Toy\Tensor\kernel_stats(true);

    $xs = [];
    foreach ($ks as $k) {
        foreach ($ns as $n) {
            $xv = [];
            for ($i = 0; $i < $k * $n; $i++) {
                $xv[] = toy_test_value($i, $n * 3 + 1);
            }
            $xs[$k][$n] = [$xv, toy_test_tensor([$k, $n], $xv)];
        }
    }

    foreach ($ms as $m) {
        foreach ($ks as $k) {
            $wv = [];
            for ($i = 0; $i < $m * $k; $i++) {
                $wv[] = toy_test_value($i, $m + $k);
            }
            $w = toy_test_tensor([$m, $k], $wv)->cast('q8_0');
            $blocks = toy_test_q8_blocks($w->toBytes());

            foreach ($ns as $n) {
                [$xv, $x] = $xs[$k][$n];
                $got = toy_test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
                [$ref, $absSums] = toy_test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
                $metrics = toy_test_q8_metrics($got, $ref, $absSums);
                $ok = $ok && $metrics['maxScaled'] < 4e-6;

                Toy\Tensor\kernel_enable(false);
                $fallback = toy_test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
                Toy\Tensor\kernel_enable(true);
                $fallbackMetrics = toy_test_q8_metrics($fallback, $got, $absSums);
                $ok = $ok && $fallbackMetrics['maxScaled'] < 1e-5;
                $ok = $ok && ($isa !== 'scalar' || $got === $fallback);
            }

            $xv = [];
            for ($i = 0; $i < $k; $i++) {
                $xv[] = toy_test_value($i, 11 + $m);
            }
            $x = toy_test_tensor([$k], $xv);
            $got = toy_test_floats(Toy\Tensor\matvec_q8_0_f32($w, $x, $m, $k));
            [$ref, $absSums] = toy_test_q8_matvec_ref($blocks, $xv, $m, $k);
            $metrics = toy_test_q8_metrics($got, $ref, $absSums);
            $ok = $ok && $metrics['maxScaled'] < 4e-6;

            Toy\Tensor\kernel_enable(false);
            $fallback = toy_test_floats(Toy\Tensor\matvec_q8_0_f32($w, $x, $m, $k));
            Toy\Tensor\kernel_enable(true);
            $fallbackMetrics = toy_test_q8_metrics($fallback, $got, $absSums);
            $ok = $ok && $fallbackMetrics['maxScaled'] < 1e-5;
            $ok = $ok && ($isa !== 'scalar' || $got === $fallback);
        }
    }

    $stats = Toy\Tensor\kernel_stats();
    $usedKernels = $usedKernels && $stats['generated'] > 0 && $stats['kernel_calls'] > 0;
}

var_dump($ok);
var_dump($usedKernels);
var_dump(count($isas) >= 1);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
done
