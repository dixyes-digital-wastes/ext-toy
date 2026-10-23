<?php
/*
 * Per-call overhead of the toy element kernels
 *
 * Run against the built module:
 *   php -n -d extension=$PWD/modules/toy.so bench/call_overhead.php
 *
 * Each case alternates rounds and reports the median of five rounds. The
 * counters at the end prove that the static calls take the frameless path and
 * the dynamic callable takes the zif path.
 */

const ROUNDS = 5;

function tensor(int $count, float $base, float $step): Toy\Tensor {
    $values = [];
    for ($i = 0; $i < $count; $i++) {
        $values[] = $base + $i * $step;
    }
    return Toy\Tensor::fromBytes('f32', [$count], pack('g*', ...$values));
}

function matrix(int $rows, int $cols): Toy\Tensor {
    $values = [];
    for ($i = 0; $i < $rows * $cols; $i++) {
        $values[] = (($i * 7) % 17) * 0.25;
    }
    return Toy\Tensor::fromBytes('f32', [$rows, $cols], pack('g*', ...$values));
}

function measure(string $label, int $iterations, callable $body): void {
    for ($i = 0; $i < 2000; $i++) {
        $body();
    }

    $samples = [];
    for ($round = 0; $round < ROUNDS; $round++) {
        $start = hrtime(true);
        for ($i = 0; $i < $iterations; $i++) {
            $body();
        }
        $samples[] = (hrtime(true) - $start) / $iterations;
    }
    sort($samples);
    printf("%-34s %9.2f ns/call\n", $label, $samples[intdiv(ROUNDS, 2)]);
}

foreach ([8, 1024, 16384] as $count) {
    $a = tensor($count, 1.0, 0.5);
    $b = tensor($count, 2.0, 0.25);
    $w = tensor($count, 3.0, 0.125);
    $iterations = $count >= 16384 ? 2000 : 20000;

    $dynamic_add = 'Toy\\Tensor\\add';
    $dynamic_scale = 'Toy\\Tensor\\scale';
    $dynamic_rmsnorm = 'Toy\\Tensor\\rmsnorm';

    printf("count = %d, %d iterations\n", $count, $iterations);

    Toy\Tensor\kernel_enable(true);
    measure('add frameless, kernels', $iterations, fn() => Toy\Tensor\add($a, $b));
    measure('add zif, kernels', $iterations, fn() => $dynamic_add($a, $b));
    measure('scale frameless, kernels', $iterations, fn() => Toy\Tensor\scale($a, 1.5));
    measure('scale zif, kernels', $iterations, fn() => $dynamic_scale($a, 1.5));
    measure('rmsnorm frameless, kernels', $iterations, fn() => Toy\Tensor\rmsnorm($a, $w));
    measure('rmsnorm zif, kernels', $iterations, fn() => $dynamic_rmsnorm($a, $w));

    Toy\Tensor\kernel_enable(false);
    measure('add frameless, scalar', $iterations, fn() => Toy\Tensor\add($a, $b));
    measure('add zif, scalar', $iterations, fn() => $dynamic_add($a, $b));
    measure('scale frameless, scalar', $iterations, fn() => Toy\Tensor\scale($a, 1.5));
    measure('rmsnorm frameless, scalar', $iterations, fn() => Toy\Tensor\rmsnorm($a, $w));
    Toy\Tensor\kernel_enable(true);
    echo "\n";
}

$left = matrix(64, 64);
$right = matrix(64, 64);
$vector = tensor(64, 1.0, 0.5);
printf("matmul 64x64x64 and matvec 64x64, 5000 iterations\n");

Toy\Tensor\kernel_enable(true);
measure('matmul zif, kernels', 5000, fn() => Toy\Tensor\matmul_f32($left, $right, 64, 64, 64));
measure('matvec zif, kernels', 5000, fn() => Toy\Tensor\matvec_f32($left, $vector, 64, 64));
Toy\Tensor\kernel_enable(false);
measure('matmul zif, scalar', 5000, fn() => Toy\Tensor\matmul_f32($left, $right, 64, 64, 64));
measure('matvec zif, scalar', 5000, fn() => Toy\Tensor\matvec_f32($left, $vector, 64, 64));
Toy\Tensor\kernel_enable(true);

echo "\n";
$stats = Toy\Tensor\kernel_stats();
printf("flf calls = %d, zif calls = %d\n", $stats['flf_calls'], $stats['zif_calls']);
printf("memo hits = %d, misses = %d\n", $stats['memo_hits'], $stats['memo_misses']);
printf("arena hits = %d, generated = %d, failures = %d\n",
    $stats['arena_hits'], $stats['generated'], $stats['generation_failures']);
printf("kernel calls = %d, scalar calls = %d, isa = %s\n",
    $stats['kernel_calls'], $stats['scalar_calls'], $stats['isa']);

/*
 * First-call cost: use element counts that were not generated above, so the
 * first call pays for codegen and the arena seal. The steady value is the
 * median of the following calls on the same tensors.
 */
echo "\nfirst call generation cost\n";
foreach ([100003, 100019, 100043] as $count) {
    $a = tensor($count, 1.0, 0.5);
    $b = tensor($count, 2.0, 0.25);

    $start = hrtime(true);
    Toy\Tensor\add($a, $b);
    $first = hrtime(true) - $start;

    $steady = [];
    for ($round = 0; $round < ROUNDS; $round++) {
        $start = hrtime(true);
        Toy\Tensor\add($a, $b);
        $steady[] = hrtime(true) - $start;
    }
    sort($steady);
    printf("count = %6d: first = %9.2f us, steady = %9.2f us, generation = %9.2f us\n",
        $count, $first / 1000.0, $steady[intdiv(ROUNDS, 2)] / 1000.0,
        ($first - $steady[intdiv(ROUNDS, 2)]) / 1000.0);
}
