<?php
declare(strict_types=1);

/*
 * Measures the accuracy and the speed of q8_0
 *
 * Usage:
 *   php -n -d extension=$PWD/modules/toy.so bench/q8_0.php
 */

require __DIR__ . '/../tests/helpers/q8.php';

ini_set('memory_limit', '512M');

function bench_best_ns(callable $fn, int $iterations): float {
    $best = INF;
    for ($i = 0; $i < $iterations; $i++) {
        $start = hrtime(true);
        $fn();
        $elapsed = hrtime(true) - $start;
        $best = min($best, (float) $elapsed);
    }
    return $best;
}

function make_matrix(int $rows, int $cols, int $seed): Toy\Tensor {
    $bytes = '';
    for ($r = 0; $r < $rows; $r++) {
        $row = [];
        for ($c = 0; $c < $cols; $c++) {
            $row[] = test_value($r * $cols + $c, $seed);
        }
        $bytes .= pack('g*', ...$row);
    }
    return Toy\Tensor::fromBytes('f32', [$rows, $cols], $bytes);
}

function make_vector(int $count, int $seed): Toy\Tensor {
    $values = [];
    for ($i = 0; $i < $count; $i++) {
        $values[] = test_value($i, $seed);
    }
    return test_tensor([$count], $values);
}

printf("PHP %s, %s\n", PHP_VERSION, Toy\Tensor\kernel_stats()['isa']);

/* ---------- accuracy ---------- */

echo "\naccuracy (double reference over dequantized weights)\n";
printf("%6s %10s %10s %10s\n", 'k', 'maxAbs', 'maxScaled', 'NMSE');
foreach ([32, 128, 1024] as $k) {
    $m = 64;
    $n = 3;
    $wv = [];
    for ($i = 0; $i < $m * $k; $i++) {
        $wv[] = test_value($i, $k);
    }
    $xv = [];
    for ($i = 0; $i < $k * $n; $i++) {
        $xv[] = test_value($i, $n * 9 + 1);
    }
    $w = test_tensor([$m, $k], $wv)->cast('q8_0');
    $x = test_tensor([$k, $n], $xv);
    $blocks = test_q8_blocks($w->toBytes());

    $got = test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
    [$ref, $absSums] = test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
    $metrics = test_q8_metrics($got, $ref, $absSums);
    printf("%6d %10.3e %10.3e %10.3e\n",
        $k, $metrics['maxAbs'], $metrics['maxScaled'], $metrics['nmse']);
}

/* ---------- matvec 4096 x 4096 ---------- */

$m = 4096;
$k = 4096;
$w32 = make_matrix($m, $k, 3);
$w8 = $w32->cast('q8_0');
$x = make_vector($k, 4);

Toy\Tensor\matvec_f32($w32, $x, $m, $k);
Toy\Tensor\matvec_q8_0_f32($w8, $x, $m, $k);

$ns32 = bench_best_ns(fn() => Toy\Tensor\matvec_f32($w32, $x, $m, $k), 3);
$ns8 = bench_best_ns(fn() => Toy\Tensor\matvec_q8_0_f32($w8, $x, $m, $k), 10);
$flops = 2.0 * $m * $k;
printf("\nmatvec %dx%d\n", $m, $k);
printf("  f32   %9.3f ms  %7.2f GFLOP/s\n", $ns32 / 1e6, $flops / $ns32);
printf("  q8_0  %9.3f ms  %7.2f GFLOP/s  (weights %d MB vs %d MB)\n",
    $ns8 / 1e6, $flops / $ns8, $w8->bytes() >> 20, $w32->bytes() >> 20);

unset($w32, $w8, $x);

/* ---------- matmul 1024 ---------- */

$m = $n = $k = 1024;
$w32 = make_matrix($m, $k, 5);
$w8 = $w32->cast('q8_0');
$x = make_matrix($k, $n, 6);

Toy\Tensor\matmul_f32($w32, $x, $m, $n, $k);
Toy\Tensor\matmul_q8_0_f32($w8, $x, $m, $n, $k);

$ns32 = bench_best_ns(fn() => Toy\Tensor\matmul_f32($w32, $x, $m, $n, $k), 3);
$ns8 = bench_best_ns(fn() => Toy\Tensor\matmul_q8_0_f32($w8, $x, $m, $n, $k), 3);
$flops = 2.0 * $m * $n * $k;
printf("\nmatmul %dx%dx%d\n", $m, $n, $k);
printf("  f32   %9.3f ms  %7.2f GFLOP/s\n", $ns32 / 1e6, $flops / $ns32);
printf("  q8_0  %9.3f ms  %7.2f GFLOP/s\n", $ns8 / 1e6, $flops / $ns8);

$stats = Toy\Tensor\kernel_stats();
printf("\ngenerated %d, kernel calls %d, scalar calls %d, isa %s, f16c %s\n",
    $stats['generated'], $stats['kernel_calls'], $stats['scalar_calls'],
    $stats['isa'], $stats['f16c_available'] ? 'yes' : 'no');
