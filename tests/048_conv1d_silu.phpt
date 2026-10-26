--TEST--
conv1d_silu applies the fused convolution and activation over row slices
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function values(Toy\Tensor $tensor): array {
    return array_values(unpack('g*', $tensor->toBytes()));
}
function conv_reference(array $w, array $x, int $rows, int $taps): array {
    $out = [];
    for ($c = 0; $c < $rows; $c++) {
        $sum = 0.0;
        for ($j = 0; $j < $taps; $j++) {
            $sum += $w[$c * $taps + $j] * $x[$j * $rows + $c];
        }
        $out[] = $sum / (1.0 + exp(-$sum));
    }
    return $out;
}
function close(float $got, float $want): bool {
    return abs($got - $want) <= 1e-6 * max(1.0, abs($want));
}

/* Channel counts that leave a rest after the eight lane activation loop */
foreach ([1, 2, 3, 7, 8, 9, 16, 17] as $rows) {
    $taps = 4;
    $w = [];
    $x = [];
    for ($i = 0; $i < $rows * $taps; $i++) {
        $w[] = 0.05 * $i - 0.4;
        $x[] = 0.03 * $i - 0.3;
    }
    $result = Toy\Tensor\conv1d_silu(tensor([$rows, $taps], $w), tensor([$taps, $rows], $x),
                                     $rows, $taps);
    $want = conv_reference($w, $x, $rows, $taps);
    $ok = $result->shape() === [$rows];
    foreach (values($result) as $i => $value) {
        $ok = $ok && close($value, $want[$i]);
    }
    var_dump($ok);
}

/* A one tap kernel is the element-wise activation of the product */
$rows = 8;
$w = [];
$x = [];
for ($i = 0; $i < $rows; $i++) {
    $w[] = 0.5 - 0.1 * $i;
    $x[] = -1.0 + 0.25 * $i;
}
$one = values(Toy\Tensor\conv1d_silu(tensor([$rows, 1], $w), tensor([1, $rows], $x), $rows, 1));
$one_ok = true;
foreach ($one as $i => $value) {
    $product = $w[$i] * $x[$i];
    $one_ok = $one_ok && close($value, $product / (1.0 + exp(-$product)));
}
var_dump($one_ok);

/* Row slicing keeps the convolution of the full call. start 5 with three
   taps moves the weights 60 bytes and the input and output 20 bytes, so the
   kernel runs on unaligned pointers */
$rows = 16;
$taps = 3;
$w = [];
$x = [];
for ($i = 0; $i < $rows * $taps; $i++) {
    $w[] = 0.02 * $i - 0.4;
    $x[] = 0.05 * $i - 0.6;
}
$weights = tensor([$rows, $taps], $w);
$inputs = tensor([$taps, $rows], $x);
$full = values(Toy\Tensor\conv1d_silu($weights, $inputs, $rows, $taps));
$slice = values(Toy\Tensor\conv1d_silu($weights, $inputs, $rows, $taps, 5, 6));
$slice_ok = $slice[0] === 0.0 && $slice[15] === 0.0;
for ($i = 5; $i < 11; $i++) {
    $slice_ok = $slice_ok && $slice[$i] === $full[$i];
}
var_dump($slice_ok);
$open = values(Toy\Tensor\conv1d_silu($weights, $inputs, $rows, $taps, 12));
$open_ok = $open[0] === 0.0 && $open[11] === 0.0;
for ($i = 12; $i < 16; $i++) {
    $open_ok = $open_ok && $open[$i] === $full[$i];
}
var_dump($open_ok);

/* out reuse */
$out = Toy\Tensor::zeros('f32', [$rows]);
var_dump(Toy\Tensor\conv1d_silu($weights, $inputs, $rows, $taps, 0, null, $out) === $out);
var_dump(values($out) === $full);

/* The generated activation is within f32 rounding of the scalar expf path */
Toy\Tensor\kernel_stats(true);
$kernel_values = values(Toy\Tensor\conv1d_silu($weights, $inputs, 16, $taps));
$stats = Toy\Tensor\kernel_stats();
$avx2 = $stats['avx2_available'];
var_dump($avx2 ? $stats['kernel_calls'] === 1 : $stats['scalar_calls'] === 1);
Toy\Tensor\kernel_enable(false);
$scalar_values = values(Toy\Tensor\conv1d_silu($weights, $inputs, 16, $taps));
Toy\Tensor\kernel_enable(true);
$same = true;
foreach ($kernel_values as $i => $value) {
    $same = $same && close($value, $scalar_values[$i]);
}
var_dump($same);
/* The target shape of one GDN layer */
$rows = 6144;
$taps = 4;
$w = array_fill(0, $rows * $taps, 0.001);
$x = array_fill(0, $rows * $taps, 0.002);
$big = values(Toy\Tensor\conv1d_silu(tensor([$rows, $taps], $w), tensor([$taps, $rows], $x),
                                     $rows, $taps));
$sum = 4 * 0.001 * 0.002;
var_dump(count($big) === $rows);
var_dump(close($big[0], $sum / (1.0 + exp(-$sum))));
var_dump(close($big[$rows - 1], $sum / (1.0 + exp(-$sum))));

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
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
