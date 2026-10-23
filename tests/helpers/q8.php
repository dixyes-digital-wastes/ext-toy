<?php
/*
 * Shared q8_0 helpers for the phpt tests
 *
 * phpize --clean deletes tests/*.php, so the helpers live in a subdirectory.
 * The reference below mirrors the C double reference: each block sums the
 * int8 products in double, then adds d * sum to the output in double.
 */

function test_tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

/* Rounds a double to the nearest f32, to mirror the C quantizer */
function test_f32(float $value): float {
    return unpack('g', pack('g', $value))[1];
}

function test_f16_to_f32(int $bits): float {
    $sign = ($bits & 0x8000) ? -1.0 : 1.0;
    $exp = ($bits >> 10) & 0x1F;
    $mant = $bits & 0x3FF;
    if ($exp === 0) {
        return $sign * $mant * 2.0 ** -24;
    }
    if ($exp === 0x1F) {
        return $mant === 0 ? $sign * INF : NAN;
    }
    return $sign * (1.0 + $mant / 1024.0) * 2.0 ** ($exp - 15);
}

function test_f32_to_f16_bits(float $value): int {
    $raw = unpack('V', pack('g', $value))[1];
    $sign = ($raw >> 16) & 0x8000;
    $exp = (int)(($raw >> 23) & 0xFF) - 127 + 15;
    $mant = $raw & 0x7FFFFF;
    if ($exp >= 0x1F) {
        return $sign | 0x7C00;
    }
    if ($exp <= 0) {
        if ($exp < -10) {
            return $sign;
        }
        $mant |= 0x800000;
        $shift = 14 - $exp;
        $half = $mant >> $shift;
        $rem = $mant & ((1 << $shift) - 1);
        $halfway = 1 << ($shift - 1);
        if ($rem > $halfway || ($rem === $halfway && ($half & 1) !== 0)) {
            $half++;
        }
        return $sign | $half;
    }
    $half = ($exp << 10) | ($mant >> 13);
    $rem = $mant & 0x1FFF;
    if ($rem > 0x1000 || ($rem === 0x1000 && ($half & 1) !== 0)) {
        $half++;
    }
    return $sign | $half;
}

function test_q8_blocks(string $bytes): array {
    $blocks = [];
    $count = intdiv(strlen($bytes), 34);
    for ($b = 0; $b < $count; $b++) {
        $raw = substr($bytes, $b * 34, 34);
        $blocks[] = [
            'd' => test_f16_to_f32(unpack('v', substr($raw, 0, 2))[1]),
            'q' => array_values(unpack('c*', substr($raw, 2))),
        ];
    }
    return $blocks;
}

/* Returns [values, absSum] where absSum bounds each dot product */
function test_q8_row_dot(array $blocks, int $first, int $blocksPerRow, array $x,
                            int $stride, int $offset): array {
    $sum = 0.0;
    $absSum = 0.0;
    for ($b = 0; $b < $blocksPerRow; $b++) {
        $block = $blocks[$first + $b];
        $s = 0.0;
        $abs = 0.0;
        for ($j = 0; $j < 32; $j++) {
            $term = $block['q'][$j] * $x[$offset + ($b * 32 + $j) * $stride];
            $s += $term;
            $abs += abs($term);
        }
        $sum += $block['d'] * $s;
        $absSum += abs($block['d']) * $abs;
    }
    return [$sum, $absSum];
}

function test_q8_matmul_ref(array $blocks, array $x, int $m, int $n, int $k): array {
    $blocksPerRow = intdiv($k, 32);
    $values = [];
    $absSums = [];
    for ($i = 0; $i < $m; $i++) {
        for ($j = 0; $j < $n; $j++) {
            [$value, $absSum] =
                test_q8_row_dot($blocks, $i * $blocksPerRow, $blocksPerRow, $x, $n, $j);
            $values[] = $value;
            $absSums[] = $absSum;
        }
    }
    return [$values, $absSums];
}

function test_q8_matvec_ref(array $blocks, array $x, int $m, int $k): array {
    $blocksPerRow = intdiv($k, 32);
    $values = [];
    $absSums = [];
    for ($i = 0; $i < $m; $i++) {
        [$value, $absSum] =
            test_q8_row_dot($blocks, $i * $blocksPerRow, $blocksPerRow, $x, 1, 0);
        $values[] = $value;
        $absSums[] = $absSum;
    }
    return [$values, $absSums];
}

function test_q8_metrics(array $got, array $ref, array $absSums): array {
    $maxAbs = 0.0;
    $maxScaled = 0.0;
    $sqErr = 0.0;
    $sqRef = 0.0;
    foreach ($ref as $i => $expected) {
        $err = abs($got[$i] - $expected);
        $maxAbs = max($maxAbs, $err);
        $maxScaled = max($maxScaled, $err / max($absSums[$i], 1e-30));
        $sqErr += $err * $err;
        $sqRef += $expected * $expected;
    }
    return [
        'maxAbs' => $maxAbs,
        'maxScaled' => $maxScaled,
        'nmse' => $sqRef > 0.0 ? $sqErr / $sqRef : 0.0,
    ];
}

/* Deterministic values with mixed signs, so sums can cancel */
function test_value(int $i, int $seed): float {
    return sin($i * 0.37 + $seed) * 0.7 + cos($i * 0.11 + $seed * 0.5) * 0.3;
}

function test_floats(Toy\Tensor $tensor): array {
    return array_values(unpack('g*', $tensor->toBytes()));
}
