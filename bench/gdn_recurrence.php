<?php
/*
 * GDN recurrence at the target shape of a 0.8B gated delta net model, against
 * the PHP double loop a transformer forward pass runs today
 *
 * Run with: php -n -d extension=$PWD/modules/toy.so bench/gdn_recurrence.php
 *
 * One token visits 18 gated delta net layers, each with 16 heads of a
 * 128 x 128 state. The PHP loop is the reference per-column recurrence with the
 * rmsnorm and the silu gate left out, so the two sides differ only in the
 * recurrence itself
 *
 * The PHP side runs at very different speeds with and without the opcache JIT,
 * so quote the baseline together with the JIT state the script prints. Both
 * runs are listed so the numbers can be reproduced:
 *
 *   php -n -d extension=$PWD/modules/toy.so bench/gdn_recurrence.php
 *   php -n -d opcache.enable_cli=1 -d opcache.jit=tracing \
 *       -d opcache.jit_buffer_size=128M -d extension=$PWD/modules/toy.so \
 *       bench/gdn_recurrence.php
 */

const HEADS = 16;
const KEY_DIM = 128;
const VALUE_DIM = 128;
const LAYERS = 18;
const ROUNDS = 5;

function values(int $seed, int $count): array {
    $state = $seed;
    $out = [];
    for ($i = 0; $i < $count; $i++) {
        $state = ($state * 1103515245 + 12345) & 0x7fffffff;
        $out[] = $state / 0x7fffffff * 2.0 - 1.0;
    }
    return $out;
}

/* One layer of the reference: q is pre-scaled, the sums run in ascending order */
function phpLayer(array $q, array $k, array $v, array $decay, array $beta, array &$state): array {
    $scale = 1.0 / sqrt((float) KEY_DIM);
    for ($i = 0; $i < HEADS * KEY_DIM; $i++) {
        $q[$i] *= $scale;
    }

    $out = [];
    for ($head = 0; $head < HEADS; $head++) {
        $stateOffset = $head * KEY_DIM * VALUE_DIM;
        $qh = array_slice($q, $head * KEY_DIM, KEY_DIM);
        $kh = array_slice($k, $head * KEY_DIM, KEY_DIM);
        $vh = array_slice($v, $head * VALUE_DIM, VALUE_DIM);
        $decayHead = $decay[$head];
        $betaHead = $beta[$head];

        for ($index = 0; $index < KEY_DIM; $index++) {
            for ($column = 0; $column < VALUE_DIM; $column++) {
                $state[$stateOffset + $index + VALUE_DIM * $column] *= $decayHead;
            }
        }

        $delta = [];
        for ($column = 0; $column < VALUE_DIM; $column++) {
            $sum = 0.0;
            for ($index = 0; $index < KEY_DIM; $index++) {
                $sum += $state[$stateOffset + $index + VALUE_DIM * $column] * $kh[$index];
            }
            $delta[$column] = $betaHead * ($vh[$column] - $sum);
        }

        for ($index = 0; $index < KEY_DIM; $index++) {
            $ki = $kh[$index];
            for ($column = 0; $column < VALUE_DIM; $column++) {
                $state[$stateOffset + $index + VALUE_DIM * $column] += $ki * $delta[$column];
            }
        }

        for ($column = 0; $column < VALUE_DIM; $column++) {
            $sum = 0.0;
            for ($index = 0; $index < KEY_DIM; $index++) {
                $sum += $state[$stateOffset + $index + VALUE_DIM * $column] * $qh[$index];
            }
            $out[$head * VALUE_DIM + $column] = $sum;
        }
    }
    return $out;
}

function tensor(string $dtype, array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes($dtype, $shape, pack('g*', ...$values));
}

function median(array $samples): float {
    sort($samples);
    return $samples[intdiv(count($samples), 2)];
}

$q = values(101, HEADS * KEY_DIM);
$k = values(202, HEADS * KEY_DIM);
$v = values(303, HEADS * VALUE_DIM);
$decay = array_map(fn(float $x): float => 0.5 + abs($x), values(404, HEADS));
$beta = array_map(fn(float $x): float => 0.25 + abs($x) * 0.75, values(505, HEADS));
$stateValues = values(606, HEADS * KEY_DIM * VALUE_DIM);
$scale = 1.0 / sqrt((float) KEY_DIM);

$tensors = [
    tensor('f32', [HEADS, KEY_DIM], $q),
    tensor('f32', [HEADS, KEY_DIM], $k),
    tensor('f32', [HEADS, VALUE_DIM], $v),
    tensor('f32', [HEADS], $decay),
    tensor('f32', [HEADS], $beta),
    tensor('f32', [HEADS, KEY_DIM, VALUE_DIM], $stateValues),
];
$outTensor = Toy\Tensor::zeros('f32', [HEADS, VALUE_DIM]);

$stats = Toy\Tensor\kernel_stats();
$isas = ['scalar'];
if ($stats['avx2_available']) {
    $isas[] = 'avx2';
}

$jit = function_exists('opcache_get_status')
    && (opcache_get_status(false)['jit']['on'] ?? false);
printf("GDN recurrence, heads=%d key_dim=%d value_dim=%d, %d layers per token\n",
    HEADS, KEY_DIM, VALUE_DIM, LAYERS);
printf("host ISA: %s, php reference: %s\n\n", implode(', ', $isas),
    $jit ? 'opcache JIT on' : 'opcache JIT off');

/* PHP reference over all layers */
$phpState = $stateValues;
$samples = [];
foreach (range(0, ROUNDS - 1) as $round) {
    $start = hrtime(true);
    for ($layer = 0; $layer < LAYERS; $layer++) {
        phpLayer($q, $k, $v, $decay, $beta, $phpState);
    }
    $samples[] = hrtime(true) - $start;
}
$phpToken = median($samples) / 1e6;
printf("%-16s %10.3f ms/token %10.3f ms/layer\n", 'php double', $phpToken, $phpToken / LAYERS);

/* One reference layer for the accuracy columns */
$phpState = $stateValues;
$phpOut = phpLayer($q, $k, $v, $decay, $beta, $phpState);

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    $samples = [];
    foreach (range(0, ROUNDS - 1) as $round) {
        $start = hrtime(true);
        for ($layer = 0; $layer < LAYERS; $layer++) {
            Toy\Tensor\gdn_recurrence($tensors[0], $tensors[1], $tensors[2], $tensors[3],
                $tensors[4], $tensors[5], 0, HEADS, 0, VALUE_DIM, $scale, $outTensor);
        }
        $samples[] = hrtime(true) - $start;
    }
    $token = median($samples) / 1e6;
    printf("%-16s %10.3f ms/token %10.3f ms/layer  %8.2fx\n",
        'toy ' . $isa, $token, $token / LAYERS, $phpToken / $token);
}

/* Accuracy and ISA agreement on one layer of a fresh state */
Toy\Tensor\kernel_isa($isas[0]);
$fresh = tensor('f32', [HEADS, KEY_DIM, VALUE_DIM], $stateValues);
Toy\Tensor\gdn_recurrence($tensors[0], $tensors[1], $tensors[2], $tensors[3],
    $tensors[4], $fresh, 0, HEADS, 0, VALUE_DIM, $scale, $outTensor);
$first = array_values(unpack('g*', $outTensor->toBytes()));

$maxScale = 0.0;
$maxDiff = 0.0;
$error = 0.0;
$energy = 0.0;
foreach ($phpOut as $i => $expected) {
    $diff = $first[$i] - $expected;
    $maxScale = max($maxScale, abs($expected));
    $maxDiff = max($maxDiff, abs($diff));
    $error += $diff * $diff;
    $energy += $expected * $expected;
}
printf("\nmatch php: maxDiff %.3e maxScaled %.3e nmse %.3e\n",
    $maxDiff, $maxDiff / $maxScale, $error / $energy);

$identical = 'n/a';
if (count($isas) > 1) {
    $scalar = tensor('f32', [HEADS, KEY_DIM, VALUE_DIM], $stateValues);
    Toy\Tensor\kernel_isa('scalar');
    Toy\Tensor\gdn_recurrence($tensors[0], $tensors[1], $tensors[2], $tensors[3],
        $tensors[4], $scalar, 0, HEADS, 0, VALUE_DIM, $scale, $outTensor);
    $scalarOut = $outTensor->toBytes();
    $scalarState = $scalar->toBytes();

    $avx2 = tensor('f32', [HEADS, KEY_DIM, VALUE_DIM], $stateValues);
    Toy\Tensor\kernel_isa('avx2');
    Toy\Tensor\gdn_recurrence($tensors[0], $tensors[1], $tensors[2], $tensors[3],
        $tensors[4], $avx2, 0, HEADS, 0, VALUE_DIM, $scale, $outTensor);
    $identical = $avx2->toBytes() === $scalarState
        && $outTensor->toBytes() === $scalarOut ? 'yes' : 'no';
}
printf("scalar and avx2 byte-identical: %s\n", $identical);
