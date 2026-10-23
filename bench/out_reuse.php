<?php
/*
 * Compare allocating a result with writing into a caller-provided out tensor.
 * Run with: php -n -d extension=$PWD/modules/toy.so bench/out_reuse.php
 */

function best_ns(callable $fn, int $iterations, int $rounds = 5): float {
    $best = INF;
    for ($round = 0; $round < $rounds; $round++) {
        $start = hrtime(true);
        for ($i = 0; $i < $iterations; $i++) {
            $fn();
        }
        $elapsed = hrtime(true) - $start;
        $best = min($best, $elapsed / $iterations);
    }
    return $best;
}

function report(string $name, float $new, float $reuse, float $alloc): void {
    $saved = $new - $reuse;
    printf(
        "%-22s new %9.0f ns  out %9.0f ns  saved %7.1f%%  zeros %8.0f ns  alloc %5.1f%%\n",
        $name,
        $new,
        $reuse,
        100.0 * $saved / $new,
        $alloc,
        100.0 * $alloc / $new
    );
}

$checksum = 0.0;

foreach ([1024, 4096, 16384] as $count) {
    $a = Toy\Tensor::full('f32', [$count], 1.0);
    $b = Toy\Tensor::full('f32', [$count], 2.0);
    $out = Toy\Tensor::zeros('f32', [$count]);
    $iterations = max(50, (int) (2000000 / $count));

    $new = best_ns(fn() => $checksum += Toy\Tensor\add($a, $b)->bytes(), $iterations);
    $reuse = best_ns(fn() => $checksum += Toy\Tensor\add($a, $b, $out)->bytes(), $iterations);
    $alloc = best_ns(fn() => $checksum += Toy\Tensor::zeros('f32', [$count])->bytes(), $iterations);
    report("add f32 {$count}", $new, $reuse, $alloc);

    $new = best_ns(fn() => $checksum += Toy\Tensor\scale($a, 2.0)->bytes(), $iterations);
    $reuse = best_ns(fn() => $checksum += Toy\Tensor\scale($a, 2.0, $out)->bytes(), $iterations);
    report("scale f32 {$count}", $new, $reuse, $alloc);
}

$m = 64;
$a = Toy\Tensor::full('f32', [$m, $m], 1.0);
$b = Toy\Tensor::full('f32', [$m, $m], 2.0);
$out = Toy\Tensor::zeros('f32', [$m, $m]);
$iterations = 400;
$new = best_ns(fn() => $checksum += Toy\Tensor\matmul_f32($a, $b, $m, $m, $m)->bytes(), $iterations);
$reuse = best_ns(fn() => $checksum += Toy\Tensor\matmul_f32($a, $b, $m, $m, $m, $out)->bytes(), $iterations);
$alloc = best_ns(fn() => $checksum += Toy\Tensor::zeros('f32', [$m, $m])->bytes(), $iterations);
report("matmul f32 {$m}x{$m}x{$m}", $new, $reuse, $alloc);

$m = 4096;
$k = 4096;
$w = Toy\Tensor::full('q8_0', [$m, $k], 1.0);
$x = Toy\Tensor::full('f32', [$k], 1.0);
$out = Toy\Tensor::zeros('f32', [$m]);
$iterations = 30;
$new = best_ns(fn() => $checksum += Toy\Tensor\matvec_q8_0_f32($w, $x, $m, $k)->bytes(), $iterations);
$reuse = best_ns(fn() => $checksum += Toy\Tensor\matvec_q8_0_f32($w, $x, $m, $k, $out)->bytes(), $iterations);
$alloc = best_ns(fn() => $checksum += Toy\Tensor::zeros('f32', [$m])->bytes(), 2000);
report("q8_0 matvec {$m}", $new, $reuse, $alloc);

fprintf(STDERR, "checksum %.0f\n", $checksum);
