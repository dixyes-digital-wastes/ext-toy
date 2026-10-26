<?php
declare(strict_types=1);

/*
 * Measures the activation quantizer on the per-token call mix of the demo
 *
 * Usage:
 *   php -n -d extension=$PWD/modules/toy.so bench/q8_quantize.php
 *
 * The demo quantizes 187 activation vectors per token: 139 of 1024 values, 24
 * of 2048 and 24 of 3584, 277504 values in total. Every worker runs the same
 * mix, so the numbers are per worker and single thread
 */

require __DIR__ . '/../tests/helpers/q8.php';

ini_set('memory_limit', '512M');

/* Widths and per-token call counts from the profile of the demo */
const TOKEN_CALLS = [
    1024 => 139,
    2048 => 24,
    3584 => 24,
];

function bench_best_ns(callable $fn, int $iterations): float {
    $best = INF;
    for ($i = 0; $i < $iterations; $i++) {
        $start = hrtime(true);
        $fn();
        $best = min($best, (float) (hrtime(true) - $start));
    }
    return $best;
}

/*
 * The PHP path that the operator replaces: Q8_0::quantize computes the
 * reciprocal and the product in double, then stores the f16 rounded scale
 */
function php_quantize(array $input): string {
    $data = '';
    $blocks = intdiv(count($input), 32);
    for ($block = 0; $block < $blocks; $block++) {
        $base = $block * 32;
        $amax = 0.0;
        for ($index = 0; $index < 32; $index++) {
            $value = abs($input[$base + $index]);
            if ($value > $amax) {
                $amax = $value;
            }
        }
        $scale = test_f32($amax / 127.0);
        $inverse = $scale !== 0.0 ? 1.0 / $scale : 0.0;
        $data .= pack('v', test_f32_to_f16_bits($scale));
        for ($index = 0; $index < 32; $index++) {
            $value = (int) round($input[$base + $index] * $inverse);
            $data .= pack('c', max(-128, min(127, $value)));
        }
    }
    return $data;
}

function count_bytes(string $left, string $right): int {
    $count = 0;
    $length = min(strlen($left), strlen($right));
    for ($i = 0; $i < $length; $i++) {
        if ($left[$i] !== $right[$i]) {
            $count++;
        }
    }
    return $count;
}

function make_vector(int $count, int $seed): Toy\Tensor {
    $values = [];
    for ($i = 0; $i < $count; $i++) {
        $values[] = test_value($i, $seed) * ($i % 7 + 1.0);
    }
    return test_tensor([$count], $values);
}

$stats = Toy\Tensor\kernel_stats();
printf("PHP %s, %s, f16c %s\n", PHP_VERSION, $stats['isa'],
    $stats['f16c_available'] ? 'yes' : 'no');

$isas = ['scalar'];
if ($stats['avx2_available']) {
    $isas[] = 'avx2';
}

/* ---------- accuracy against the byte references ---------- */

echo "\naccuracy (bytes of one 1024 value vector)\n";
$probe = make_vector(1024, 5);
$probeValues = test_floats($probe);
$model = test_q8_quantize($probeValues);
$php = php_quantize($probeValues);
printf("  f32 rule reference %s\n", $probe->cast('q8_0')->toBytes() === $model ? 'match' : 'MISMATCH');
printf("  f32 reference via quantize_q8_0 %s\n",
    Toy\Tensor\quantize_q8_0($probe)->toBytes() === $model ? 'match' : 'MISMATCH');

/* ---------- per width ---------- */

echo "\nper call, best of 20\n";
printf("%6s %8s %12s %12s %12s %10s %10s\n",
    'width', 'calls', 'php us', 'scalar us', 'avx2 us', 'scalar x', 'avx2 x');

$total = ['php' => 0.0, 'scalar' => 0.0, 'avx2' => 0.0];
$diffValues = 0;
$diffValuesPhp = 0;
$phpBytes = 0;
foreach (TOKEN_CALLS as $width => $calls) {
    $tensor = make_vector($width, 11);
    $values = test_floats($tensor);
    $reference = test_q8_quantize($values);

    $php = php_quantize($values);
    $phpNs = bench_best_ns(fn () => php_quantize($values), 5);
    $scalarNs = 0.0;
    $avx2Ns = 0.0;
    foreach ($isas as $isa) {
        Toy\Tensor\kernel_isa($isa);
        Toy\Tensor\kernel_stats(true);
        $ns = bench_best_ns(fn () => Toy\Tensor\quantize_q8_0($tensor), 20);
        if ($isa === 'scalar') {
            $scalarNs = $ns;
        } else {
            $avx2Ns = $ns;
        }
        $bytes = Toy\Tensor\quantize_q8_0($tensor)->toBytes();
        $diffValues += count_bytes($bytes, $reference);
        // The double precision PHP variant, for the record
        if ($isa === $isas[count($isas) - 1]) {
            $diffValuesPhp += count_bytes($bytes, $php);
            $phpBytes += strlen($php);
        }
    }
    if ($avx2Ns === 0.0) {
        $avx2Ns = $scalarNs;
    }

    $total['php'] += $phpNs * $calls;
    $total['scalar'] += $scalarNs * $calls;
    $total['avx2'] += $avx2Ns * $calls;

    printf("%6d %8d %12.2f %12.2f %12.2f %10.2f %10.2f\n",
        $width, $calls, $phpNs / 1e3, $scalarNs / 1e3, $avx2Ns / 1e3,
        $phpNs / $scalarNs, $phpNs / $avx2Ns);
}

/* ---------- per token ---------- */

$values = array_sum(array_map(fn (int $width, int $calls): int => $width * $calls,
    array_keys(TOKEN_CALLS), array_values(TOKEN_CALLS)));
$calls = array_sum(TOKEN_CALLS);

printf("\nper token: %d calls, %d values, single thread\n", $calls, $values);
printf("  php    %8.3f ms\n", $total['php'] / 1e6);
printf("  scalar %8.3f ms  %6.1f x vs php  %7.2f Mvalue/s\n",
    $total['scalar'] / 1e6, $total['php'] / $total['scalar'], $values / ($total['scalar'] / 1e9) / 1e6);
printf("  avx2   %8.3f ms  %6.1f x vs php  %7.2f Mvalue/s  %5.2f x vs scalar\n",
    $total['avx2'] / 1e6, $total['php'] / $total['avx2'],
    $values / ($total['avx2'] / 1e9) / 1e6, $total['scalar'] / $total['avx2']);

printf("\nbytes that differ from the f32 rule reference: %d\n", $diffValues);
printf("bytes that differ from the double precision PHP variant: %d of %d\n",
    $diffValuesPhp, $phpBytes);

/* ---------- rows ---------- */

$rows = 128;
$width = 1024;
$rowValues = [];
for ($i = 0; $i < $rows * $width; $i++) {
    $rowValues[] = test_value($i, 3) * ($i % 5 + 1.0);
}
$matrix = test_tensor([$rows, $width], $rowValues);
$half = intdiv($rows, 2);

echo "\nrows and slices, {$rows} x {$width}\n";
foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    $whole = bench_best_ns(fn () => Toy\Tensor\quantize_q8_0($matrix), 10);
    $range = bench_best_ns(fn () => Toy\Tensor\quantize_q8_0($matrix, $half, $rows), 10);
    printf("  %-6s whole %8.1f us  half rows %8.1f us (%.2f of the whole)\n",
        $isa, $whole / 1e3, $range / 1e3, $range / $whole);
}

/* ---------- kernels ---------- */

$final = Toy\Tensor\kernel_stats();
printf("\ngenerated %d, kernel calls %d, scalar calls %d, isa %s\n",
    $final['generated'], $final['kernel_calls'], $final['scalar_calls'], $final['isa']);
