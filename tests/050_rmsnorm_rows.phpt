--TEST--
rmsnorm_rows normalizes every row against one weight row, including slices
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
function row_reference(array $values, array $weight, int $rows, int $width, float $eps): array {
    $out = [];
    for ($r = 0; $r < $rows; $r++) {
        $squares = 0.0;
        for ($i = 0; $i < $width; $i++) {
            $squares += $values[$r * $width + $i] * $values[$r * $width + $i];
        }
        $scale = 1.0 / sqrt($squares / $width + $eps);
        for ($i = 0; $i < $width; $i++) {
            $out[] = $values[$r * $width + $i] * $scale * $weight[$i];
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
    $weight = [];
    for ($i = 0; $i < $rows * $width; $i++) {
        $input[] = 0.05 * $i - 1.0;
    }
    for ($i = 0; $i < $width; $i++) {
        $weight[] = 0.5 + 0.25 * ($i % 5);
    }
    $x = tensor([$rows, $width], $input);
    $w = tensor([$width], $weight);
    $result = Toy\Tensor\rmsnorm_rows($x, $w);
    $want = row_reference($input, $weight, $rows, $width, 1e-6);
    $ok = $result->shape() === [$rows, $width];
    foreach (values($result) as $i => $value) {
        $ok = $ok && close($value, $want[$i]);
    }
    var_dump($ok);
}

$width = 5;
$input = [];
$weight = [];
for ($i = 0; $i < $rows * $width; $i++) {
    $input[] = 0.05 * $i - 1.0;
}
for ($i = 0; $i < $width; $i++) {
    $weight[] = 0.5 + 0.25 * ($i % 5);
}
$x = tensor([$rows, $width], $input);
$w = tensor([$width], $weight);
$full = values(Toy\Tensor\rmsnorm_rows($x, $w));

/* A row slice writes only its rows. start 1 with width 5 moves the row
   pointers 20 bytes, so the vector loads and stores are not aligned */
$slice = values(Toy\Tensor\rmsnorm_rows($x, $w, 1e-6, 1, 1));
$slice_ok = $slice[0] === 0.0 && $slice[4] === 0.0 && $slice[10] === 0.0;
for ($i = 5; $i < 10; $i++) {
    $slice_ok = $slice_ok && $slice[$i] === $full[$i];
}
var_dump($slice_ok);
$open = values(Toy\Tensor\rmsnorm_rows($x, $w, 1e-6, 2));
var_dump($open[0] === 0.0 && $open[10] === $full[10] && $open[14] === $full[14]);

/* eps moves into the square root, and the weight may be zero or negative */
foreach ([0.0, 1e-6, 1.0] as $eps) {
    $got = values(Toy\Tensor\rmsnorm_rows($x, $w, $eps));
    $want = row_reference($input, $weight, $rows, $width, $eps);
    $ok = true;
    foreach ($got as $i => $value) {
        $ok = $ok && close($value, $want[$i]);
    }
    var_dump($ok);
}
$signed = tensor([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
$signed_weight = tensor([3], [1.0, 0.0, -2.0]);
echo json_encode(array_map(fn(float $v): float => round($v, 5),
    values(Toy\Tensor\rmsnorm_rows($signed, $signed_weight, 0.0)))), "\n";

/* The head shapes of the demo: 16 heads of 128 and 8 heads of 256 */
$head = tensor([16, 128], array_fill(0, 2048, 0.5));
$head_weight = tensor([128], array_fill(0, 128, 2.0));
$head_out = values(Toy\Tensor\rmsnorm_rows($head, $head_weight));
var_dump(close($head_out[0], 0.5 * (1.0 / sqrt(0.25 + 1e-6)) * 2.0));
$heads8 = tensor([8, 256], array_fill(0, 2048, -0.25));
$weight256 = tensor([256], array_fill(0, 256, 0.5));
var_dump(count(values(Toy\Tensor\rmsnorm_rows($heads8, $weight256))) === 2048);

/* out reuse and in place */
$out = Toy\Tensor::zeros('f32', [$rows, $width]);
var_dump(Toy\Tensor\rmsnorm_rows($x, $w, 1e-6, 0, null, $out) === $out);
var_dump($out->toBytes() === Toy\Tensor\rmsnorm_rows($x, $w)->toBytes());
$in_place = tensor([$rows, $width], $input);
var_dump(Toy\Tensor\rmsnorm_rows($in_place, $w, 1e-6, 0, null, $in_place) === $in_place);
var_dump($in_place->toBytes() === $out->toBytes());

$rmsnorm_rows = new ReflectionFunction('Toy\\Tensor\\rmsnorm_rows');
var_dump($rmsnorm_rows->getNumberOfParameters() === 6);
var_dump($rmsnorm_rows->getNumberOfRequiredParameters() === 2);

/* The f32 sum of squares is the same chain on both paths */
Toy\Tensor\kernel_stats(true);
$kernel_values = values(Toy\Tensor\rmsnorm_rows($x, $w));
$stats = Toy\Tensor\kernel_stats();
$avx2 = $stats['avx2_available'];
var_dump($avx2 ? $stats['kernel_calls'] === 1 : $stats['scalar_calls'] === 1);
Toy\Tensor\kernel_enable(false);
$scalar_values = values(Toy\Tensor\rmsnorm_rows($x, $w));
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
[0.46291,0,-2.77746,0.78954,0,-2.36863]
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
