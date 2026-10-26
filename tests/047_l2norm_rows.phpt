--TEST--
l2norm_rows normalizes every row, including tails, eps values and row slices
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
function row_reference(array $values, int $rows, int $width, float $eps): array {
    $out = [];
    for ($r = 0; $r < $rows; $r++) {
        $squares = 0.0;
        for ($i = 0; $i < $width; $i++) {
            $squares += $values[$r * $width + $i] * $values[$r * $width + $i];
        }
        $scale = 1.0 / sqrt($squares + $eps);
        for ($i = 0; $i < $width; $i++) {
            $out[] = $values[$r * $width + $i] * $scale;
        }
    }
    return $out;
}
function close(float $got, float $want): bool {
    return abs($got - $want) <= 1e-6 * max(1.0, abs($want));
}

/* Widths that leave a rest after the eight element output loop */
$rows = 3;
foreach ([1, 5, 7, 8, 9, 128] as $width) {
    $input = [];
    for ($i = 0; $i < $rows * $width; $i++) {
        $input[] = 0.05 * $i - 1.0;
    }
    $x = tensor([$rows, $width], $input);
    $result = Toy\Tensor\l2norm_rows($x);
    $want = row_reference($input, $rows, $width, 1e-6);
    $ok = $result->shape() === [$rows, $width];
    foreach (values($result) as $i => $value) {
        $ok = $ok && close($value, $want[$i]);
    }
    var_dump($ok);
}

$width = 5;
$input = [];
for ($i = 0; $i < $rows * $width; $i++) {
    $input[] = 0.05 * $i - 1.0;
}
$x = tensor([$rows, $width], $input);
$full = values(Toy\Tensor\l2norm_rows($x));

/* A row slice writes only its rows. start 1 with width 5 moves the row
   pointers 20 bytes, so the vector loads and stores are not aligned */
$slice = values(Toy\Tensor\l2norm_rows($x, 1e-6, 1, 1));
$slice_ok = $slice[0] === 0.0 && $slice[4] === 0.0 && $slice[10] === 0.0;
for ($i = 5; $i < 10; $i++) {
    $slice_ok = $slice_ok && $slice[$i] === $full[$i];
}
var_dump($slice_ok);

/* The default count covers the rows from start on */
$open = values(Toy\Tensor\l2norm_rows($x, 1e-6, 2));
var_dump($open[0] === 0.0 && $open[10] === $full[10] && $open[14] === $full[14]);

/* eps moves into the square root */
foreach ([0.0, 1e-6, 1.0] as $eps) {
    $got = values(Toy\Tensor\l2norm_rows($x, $eps));
    $want = row_reference($input, $rows, $width, $eps);
    $ok = true;
    foreach ($got as $i => $value) {
        $ok = $ok && close($value, $want[$i]);
    }
    var_dump($ok);
}

/* Exact cases: one value, and a row whose norm is an integer */
var_dump(values(Toy\Tensor\l2norm_rows(tensor([1, 1], [2.0]), 0.0))[0] === 1.0);
var_dump(values(Toy\Tensor\l2norm_rows(tensor([1, 4], [3.0, 4.0, 0.0, 0.0]), 0.0))[0]
    === 0.6000000238418579);

/* out reuse and in place */
$out = Toy\Tensor::zeros('f32', [$rows, $width]);
var_dump(Toy\Tensor\l2norm_rows($x, 1e-6, 0, null, $out) === $out);
var_dump($out->toBytes() === Toy\Tensor\l2norm_rows($x)->toBytes());
$in_place = tensor([$rows, $width], $input);
var_dump(Toy\Tensor\l2norm_rows($in_place, 1e-6, 0, null, $in_place) === $in_place);
var_dump($in_place->toBytes() === $out->toBytes());

/* The f32 sum of squares is the same chain on both paths */
Toy\Tensor\kernel_stats(true);
$kernel_values = values(Toy\Tensor\l2norm_rows($x));
$stats = Toy\Tensor\kernel_stats();
$avx2 = $stats['avx2_available'];
var_dump($avx2 ? $stats['kernel_calls'] === 1 : $stats['scalar_calls'] === 1);
Toy\Tensor\kernel_enable(false);
$scalar_values = values(Toy\Tensor\l2norm_rows($x));
Toy\Tensor\kernel_enable(true);
var_dump($scalar_values === $kernel_values);
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
bool(true)
done
