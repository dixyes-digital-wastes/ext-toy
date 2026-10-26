<?php
declare(strict_types=1);

/*
 * Measures the math operators at the shapes of the demo model, and the state
 * shift of the convolution
 *
 * Every case of the first table runs the generated AVX2 kernel and then the
 * scalar C reference, so the speedup column is the kernel against the
 * reference on one thread. The last column projects one token through 18 GDN
 * layers. The second table compares shift_rows with the PHP array shift of
 * the demo, which copies the same values
 *
 * Usage:
 *   php -n -d extension=$PWD/modules/toy.so bench/ops.php
 */

function bench_ns(callable $fn, int $iterations): float {
    $fn();
    $best = INF;
    for ($i = 0; $i < $iterations; $i++) {
        $start = hrtime(true);
        $fn();
        $elapsed = (float) (hrtime(true) - $start);
        $best = min($best, $elapsed);
    }
    return $best;
}

function make_tensor(array $shape, callable $value): Toy\Tensor {
    $count = array_product($shape);
    $values = [];
    for ($i = 0; $i < $count; $i++) {
        $values[] = $value($i);
    }
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

$stats = Toy\Tensor\kernel_stats();
printf("PHP %s, codegen ISA %s, AVX2 %s\n\n", PHP_VERSION, $stats['isa'],
    $stats['avx2_available'] ? 'available' : 'missing');

if (!$stats['avx2_available']) {
    echo "the host has no AVX2, the kernel and the reference are the same code\n";
    exit(0);
}

/* The shape of the demo model: 18 GDN layers, 16 heads of 128 values each */
$gdn_layers = 18;
$sig_input = make_tensor([2048], fn(int $i): float => ($i % 13) * 0.25 - 1.5);
$sp_input = make_tensor([2048], fn(int $i): float => ($i % 11) * 0.5 - 2.5);
$norm_input = make_tensor([16, 128], fn(int $i): float => (($i * 37) % 100) / 100.0 - 0.5);
$norm_weight = make_tensor([128], fn(int $i): float => 0.5 + 0.01 * ($i % 7));
$conv_rows = 6144;
$conv_taps = 4;
$conv_weights = make_tensor([$conv_rows, $conv_taps],
    fn(int $i): float => (($i * 29) % 100) / 100.0 - 0.5);
$conv_input = make_tensor([$conv_taps, $conv_rows],
    fn(int $i): float => (($i * 17) % 100) / 100.0 - 0.5);

$cases = [
    ['sigmoid', '2048', fn(): Toy\Tensor => Toy\Tensor\sigmoid($sig_input), 1],
    ['softplus', '2048', fn(): Toy\Tensor => Toy\Tensor\softplus($sp_input), 1],
    ['l2norm_rows', '16x128', fn(): Toy\Tensor => Toy\Tensor\l2norm_rows($norm_input),
        $gdn_layers],
    ['rmsnorm_rows', '16x128',
        fn(): Toy\Tensor => Toy\Tensor\rmsnorm_rows($norm_input, $norm_weight),
        $gdn_layers],
    ['conv1d_silu', '6144x4',
        fn(): Toy\Tensor => Toy\Tensor\conv1d_silu($conv_weights, $conv_input, $conv_rows,
            $conv_taps), $gdn_layers],
];

printf("%-14s %-8s %10s %11s %9s %11s\n", 'op', 'shape', 'kernel ns', 'scalar ns', 'speedup',
    'ms/token');
$kernel_total = 0.0;
$scalar_total = 0.0;
foreach ($cases as [$name, $shape, $run, $layers]) {
    Toy\Tensor\kernel_enable(true);
    $kernel_ns = bench_ns($run, 50);
    Toy\Tensor\kernel_enable(false);
    $scalar_ns = bench_ns($run, 50);
    Toy\Tensor\kernel_enable(true);
    printf("%-14s %-8s %10.1f %11.1f %8.2fx %11.3f\n", $name, $shape, $kernel_ns, $scalar_ns,
        $scalar_ns / $kernel_ns, $kernel_ns * $layers / 1e6);
    $kernel_total += $kernel_ns * $layers;
    $scalar_total += $scalar_ns * $layers;
}

printf("\nGDN math of one token over 18 layers: %.3f ms kernel, %.3f ms scalar, %.2fx\n",
    $kernel_total / 1e6, $scalar_total / 1e6, $scalar_total / $kernel_total);

/*
 * The state of one convolution layer is four rows of 6144 channels with the
 * newest step last. A token shifts the rows up and writes the new values into
 * the last row
 */
$state_rows = 6144;
$state_taps = 4;
$state = make_tensor([$state_taps, $state_rows], fn(int $i): float => (($i * 13) % 100) / 100.0);
$state_row = make_tensor([$state_rows], fn(int $i): float => (($i * 7) % 100) / 100.0);
$shift = fn(): Toy\Tensor => Toy\Tensor\shift_rows($state, $state_row, $state);

/* The demo keeps the history in a PHP array and merges the new row */
$history = array_slice(array_values(unpack('g*', $state->toBytes())), 0,
    ($state_taps - 1) * $state_rows);
$qkv = array_values(unpack('g*', $state_row->toBytes()));
$php_shift = function () use (&$history, $qkv, $state_rows): void {
    $history = array_merge(array_slice($history, $state_rows), $qkv);
};

printf("\nstate update of one layer, [%d, %d]:\n", $state_taps, $state_rows);
printf("%-14s %-8s %10s %11s %9s %11s\n", 'op', 'shape', 'op ns', 'php ns', 'ratio',
    'ms/token');
$shift_ns = bench_ns($shift, 50);
$php_ns = bench_ns($php_shift, 20);
printf("%-14s %-8s %10.1f %11.1f %8.2fx %11.3f\n", 'shift_rows', '4x6144', $shift_ns,
    $php_ns, $php_ns / $shift_ns, $shift_ns * $gdn_layers / 1e6);
