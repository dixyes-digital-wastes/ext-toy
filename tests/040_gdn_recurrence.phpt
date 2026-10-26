--TEST--
gdn_recurrence matches the bit-exact column reference for sliced heads and columns
--EXTENSIONS--
toy
--FILE--
<?php
/*
 * The reference runs the same f32 sequence as the kernel: every product enters
 * through a single-rounded FMA and every other step rounds once, in the
 * per-column order the operator documents
 */
function f32(float $value): float {
    return unpack('g', pack('g', $value))[1];
}
function fma(float $a, float $b, float $c): float {
    // a * b is exact in double for f32 inputs, so one rounding remains
    return f32($a * $b + $c);
}
function unit(int $seed, int $count): array {
    $values = [];
    $state = $seed;
    for ($i = 0; $i < $count; $i++) {
        $state = ($state * 1103515245 + 12345) & 0x7fffffff;
        $values[] = f32($state / 0x7fffffff * 2.0 - 1.0);
    }
    return $values;
}
function range_values(int $seed, int $count, float $low, float $high): array {
    $values = [];
    $state = $seed;
    for ($i = 0; $i < $count; $i++) {
        $state = ($state * 1103515245 + 12345) & 0x7fffffff;
        $values[] = f32($low + $state / 0x7fffffff * ($high - $low));
    }
    return $values;
}
function make_case(int $seed, int $heads, int $keyDim, int $valueDim,
                   int $headBegin, int $headEnd, int $colBegin, int $colEnd,
                   float $scale): array {
    return [
        'heads' => $heads, 'keyDim' => $keyDim, 'valueDim' => $valueDim,
        'headBegin' => $headBegin, 'headEnd' => $headEnd,
        'colBegin' => $colBegin, 'colEnd' => $colEnd, 'scale' => $scale,
        'q' => unit($seed + 1, $heads * $keyDim),
        'k' => unit($seed + 2, $heads * $keyDim),
        'v' => unit($seed + 3, $heads * $valueDim),
        'decay' => range_values($seed + 4, $heads, 0.5, 1.0),
        'beta' => range_values($seed + 5, $heads, 0.25, 1.0),
        'state' => unit($seed + 6, $heads * $keyDim * $valueDim),
    ];
}
/* Lane fold of the dots: ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7)) */
function fold(array $lane): float {
    $even = f32(f32($lane[0] + $lane[4]) + f32($lane[2] + $lane[6]));
    $odd = f32(f32($lane[1] + $lane[5]) + f32($lane[3] + $lane[7]));
    return f32($even + $odd);
}
/* Returns the updated state and the outputs, both as packed f32 bytes */
function reference(array $c): array {
    $q = $c['q'];
    $k = $c['k'];
    $v = $c['v'];
    $decay = $c['decay'];
    $beta = $c['beta'];
    $state = $c['state'];
    $keyDim = $c['keyDim'];
    $valueDim = $c['valueDim'];
    $key8 = intdiv($keyDim, 8) * 8;
    $scale = f32($c['scale']);
    $out = array_fill(0, $c['heads'] * $valueDim, 0.0);

    for ($head = $c['headBegin']; $head < $c['headEnd']; $head++) {
        $qBase = $head * $keyDim;
        $vBase = $head * $valueDim;
        $stateBase = $head * $valueDim * $keyDim;
        for ($col = $c['colBegin']; $col < $c['colEnd']; $col++) {
            $row = $stateBase + $col * $keyDim;
            $lane = array_fill(0, 8, 0.0);
            $i = 0;
            for (; $i < $key8; $i += 8) {
                for ($l = 0; $l < 8; $l++) {
                    $s = f32($state[$row + $i + $l] * $decay[$head]);
                    $state[$row + $i + $l] = $s;
                    $lane[$l] = fma($s, $k[$qBase + $i + $l], $lane[$l]);
                }
            }
            $sum = fold($lane);
            for (; $i < $keyDim; $i++) {
                $s = f32($state[$row + $i] * $decay[$head]);
                $state[$row + $i] = $s;
                $sum = fma($s, $k[$qBase + $i], $sum);
            }

            $delta = f32($beta[$head] * f32($v[$vBase + $col] - $sum));

            $lane = array_fill(0, 8, 0.0);
            $i = 0;
            for (; $i < $key8; $i += 8) {
                for ($l = 0; $l < 8; $l++) {
                    $s = fma($k[$qBase + $i + $l], $delta, $state[$row + $i + $l]);
                    $state[$row + $i + $l] = $s;
                    $lane[$l] = fma($s, f32($q[$qBase + $i + $l] * $scale), $lane[$l]);
                }
            }
            $result = fold($lane);
            for (; $i < $keyDim; $i++) {
                $s = fma($k[$qBase + $i], $delta, $state[$row + $i]);
                $state[$row + $i] = $s;
                $result = fma($s, f32($q[$qBase + $i] * $scale), $result);
            }
            $out[$vBase + $col] = $result;
        }
    }
    return [pack('g*', ...$state), pack('g*', ...$out)];
}
function tensors(array $c): array {
    return [
        Toy\Tensor::fromBytes('f32', [$c['heads'], $c['keyDim']], pack('g*', ...$c['q'])),
        Toy\Tensor::fromBytes('f32', [$c['heads'], $c['keyDim']], pack('g*', ...$c['k'])),
        Toy\Tensor::fromBytes('f32', [$c['heads'], $c['valueDim']], pack('g*', ...$c['v'])),
        Toy\Tensor::fromBytes('f32', [$c['heads']], pack('g*', ...$c['decay'])),
        Toy\Tensor::fromBytes('f32', [$c['heads']], pack('g*', ...$c['beta'])),
        Toy\Tensor::fromBytes('f32', [$c['heads'], $c['valueDim'], $c['keyDim']],
            pack('g*', ...$c['state'])),
    ];
}
function run(array $c, array $t, int $headBegin, int $headEnd, int $colBegin, int $colEnd,
             ?Toy\Tensor $out = null): Toy\Tensor {
    return Toy\Tensor\gdn_recurrence($t[0], $t[1], $t[2], $t[3], $t[4], $t[5],
        $headBegin, $headEnd, $colBegin, $colEnd, $c['scale'], $out);
}

$stats = Toy\Tensor\kernel_stats();
$isas = ['scalar'];
if ($stats['avx2_available']) {
    $isas[] = 'avx2';
}

$cases = [
    // single head and a single column
    [11, 1, 1, 1, 0, 1, 0, 1, 1.0],
    // one head, a value width of two blocks plus one scalar column
    [12, 1, 5, 17, 0, 1, 0, 17, 0.5],
    // several heads, a column slice that starts and ends unaligned
    [13, 3, 7, 10, 0, 3, 2, 9, 0.75],
    // a head slice of a larger tensor
    [14, 2, 3, 5, 1, 2, 0, 5, 1.25],
    // an unaligned start with a short tail
    [15, 4, 3, 8, 0, 4, 3, 8, 0.25],
    // a key width of one block plus a tail, with a head and a column slice
    [18, 2, 13, 9, 1, 2, 2, 7, 0.75],
    // empty column and head ranges leave the state alone
    [16, 2, 4, 6, 0, 2, 0, 0, 0.5],
    [17, 2, 4, 6, 2, 2, 0, 6, 0.5],
];

$valuesMatch = 0;
$avx2Match = 0;
$splitMatch = 0;
$casesRun = 0;

foreach ($cases as [$seed, $heads, $keyDim, $valueDim,
                   $headBegin, $headEnd, $colBegin, $colEnd, $scale]) {
    $case = make_case($seed, $heads, $keyDim, $valueDim, $headBegin, $headEnd,
        $colBegin, $colEnd, $scale);
    [$refState, $refOut] = reference($case);
    $stateBefore = $case['state'];

    foreach ($isas as $isa) {
        Toy\Tensor\kernel_isa($isa);
        $t = tensors($case);
        $outValue = Toy\Tensor::zeros('f32', [$heads, $valueDim]);
        $result = run($case, $t, $headBegin, $headEnd, $colBegin, $colEnd, $outValue);
        $casesRun++;

        // out is reused, the state is updated in place, both match the reference
        $matches = $result === $outValue
            && $t[5]->toBytes() === $refState
            && $outValue->toBytes() === $refOut;
        if ($headBegin === $headEnd || $colBegin === $colEnd) {
            $matches = $matches && $t[5]->toBytes() === pack('g*', ...$stateBefore)
                && $outValue->toBytes() === str_repeat("\0", $heads * $valueDim * 4);
        } else {
            $matches = $matches && $t[5]->toBytes() !== pack('g*', ...$stateBefore);
        }
        if ($isa === 'scalar') {
            $valuesMatch += (int) $matches;
        } else {
            $avx2Match += (int) $matches;
        }
    }

    // head and column shards of the same range give the same bytes
    Toy\Tensor\kernel_isa('scalar');
    $full = tensors($case);
    $fullOut = run($case, $full, $headBegin, $headEnd, $colBegin, $colEnd);

    $sharded = tensors($case);
    $shardedOut = Toy\Tensor::zeros('f32', [$heads, $valueDim]);
    for ($head = $headBegin; $head < $headEnd; $head++) {
        run($case, $sharded, $head, $head + 1, $colBegin, $colEnd, $shardedOut);
    }
    $width = $colEnd - $colBegin;
    $cuts = [$colBegin, $colBegin + intdiv($width, 3), $colBegin + intdiv(2 * $width, 3), $colEnd];
    $columns = tensors($case);
    $columnsOut = Toy\Tensor::zeros('f32', [$heads, $valueDim]);
    for ($i = 0; $i + 1 < count($cuts); $i++) {
        if ($cuts[$i] < $cuts[$i + 1]) {
            run($case, $columns, $headBegin, $headEnd, $cuts[$i], $cuts[$i + 1], $columnsOut);
        }
    }
    $splitMatch += (int) ($sharded[5]->toBytes() === $full[5]->toBytes()
        && $shardedOut->toBytes() === $fullOut->toBytes()
        && $columns[5]->toBytes() === $full[5]->toBytes()
        && $columnsOut->toBytes() === $fullOut->toBytes());
}

var_dump($valuesMatch === count($cases));
var_dump($avx2Match === (count($isas) > 1 ? count($cases) : 0));
var_dump($splitMatch === count($cases));
var_dump($casesRun > 0);

// The target shape only compares the two ISAs; the PHP model would be slow
$target = make_case(21, 16, 128, 128, 0, 16, 0, 128, 1.0 / sqrt(128.0));
$scalar = tensors($target);
Toy\Tensor\kernel_isa('scalar');
$scalarOut = run($target, $scalar, 0, 16, 0, 128);
$scalarState = $scalar[5]->toBytes();
$targetMatch = true;
if ($stats['avx2_available']) {
    Toy\Tensor\kernel_isa('avx2');
    $avx2 = tensors($target);
    $avx2Out = run($target, $avx2, 0, 16, 0, 128);
    $targetMatch = $avx2Out->toBytes() === $scalarOut->toBytes()
        && $avx2[5]->toBytes() === $scalarState;
}
var_dump($targetMatch);

// The counters show which path ran
$case = make_case(31, 2, 4, 6, 0, 2, 0, 6, 0.5);
Toy\Tensor\kernel_isa('scalar');
Toy\Tensor\kernel_stats(true);
$t = tensors($case);
run($case, $t, 0, 2, 0, 6);
$scalarStats = Toy\Tensor\kernel_stats();
$dispatchMatch = $scalarStats['scalar_calls'] >= 1 && $scalarStats['kernel_calls'] === 0;

if ($stats['avx2_available']) {
    Toy\Tensor\kernel_isa('avx2');
    Toy\Tensor\kernel_stats(true);
    $t = tensors($case);
    run($case, $t, 0, 2, 0, 6);
    $avx2Stats = Toy\Tensor\kernel_stats();
    $dispatchMatch = $dispatchMatch
        && $avx2Stats['kernel_calls'] >= 1 && $avx2Stats['scalar_calls'] === 0;

    // kernel_enable(false) keeps the scalar path even when the ISA is avx2
    Toy\Tensor\kernel_enable(false);
    $t = tensors($case);
    run($case, $t, 0, 2, 0, 6);
    $disabledStats = Toy\Tensor\kernel_stats();
    Toy\Tensor\kernel_enable(true);
    $dispatchMatch = $dispatchMatch
        && $disabledStats['kernel_calls'] === $avx2Stats['kernel_calls']
        && $disabledStats['scalar_calls'] === $avx2Stats['scalar_calls'] + 1;
}
var_dump($dispatchMatch);

// The default scale is one and the default out is a new tensor
$c = make_case(41, 2, 4, 6, 0, 2, 0, 6, 1.0);
$t = tensors($c);
Toy\Tensor\kernel_isa('scalar');
$default = Toy\Tensor\gdn_recurrence($t[0], $t[1], $t[2], $t[3], $t[4], $t[5], 0, 2, 0, 6);
$t = tensors($c);
$explicit = Toy\Tensor\gdn_recurrence($t[0], $t[1], $t[2], $t[3], $t[4], $t[5], 0, 2, 0, 6, 1.0);
var_dump($default->toBytes() === $explicit->toBytes());
var_dump($default->shape() === [2, 6]);

$function = new ReflectionFunction('Toy\\Tensor\\gdn_recurrence');
var_dump($function->getNumberOfParameters() === 12);
var_dump($function->getNumberOfRequiredParameters() === 10);
var_dump($function->getParameters()[10]->getDefaultValue() === 1.0);
var_dump($function->getParameters()[11]->getDefaultValue() === null);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
