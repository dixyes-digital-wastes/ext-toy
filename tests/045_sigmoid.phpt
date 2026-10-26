--TEST--
sigmoid matches the reference within f32 tolerance and reuses an out tensor
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
function reference(float $value): float {
    return 1.0 / (1.0 + exp(-$value));
}
function close(float $got, float $want): bool {
    return abs($got - $want) <= 1e-6 * max(1.0, abs($want));
}

$inputs = [-20.0, -8.0, -3.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 3.0, 8.0, 20.0];
$single = tensor([count($inputs)], $inputs);
$result = Toy\Tensor\sigmoid($single);
echo $result->dtype, ' ', json_encode($result->shape()), "\n";

$all_close = true;
foreach (values($result) as $i => $value) {
    $all_close = $all_close && close($value, reference($inputs[$i]));
}
var_dump($all_close);

$out = Toy\Tensor::zeros('f32', [count($inputs)]);
var_dump(Toy\Tensor\sigmoid($single, $out) === $out);
var_dump($out->toBytes() === $result->toBytes());

$in_place = tensor([count($inputs)], $inputs);
var_dump(Toy\Tensor\sigmoid($in_place, $in_place) === $in_place);
var_dump($in_place->toBytes() === $result->toBytes());
var_dump(Toy\Tensor\sigmoid($single, null)->toBytes() === $result->toBytes());

/* Tails: every count from 1 to 17 covers the eight element loop and its rest */
$tails = true;
for ($count = 1; $count <= 17; $count++) {
    $input = [];
    for ($i = 0; $i < $count; $i++) {
        $input[] = ($i % 7) * 0.5 - 1.5;
    }
    $got = values(Toy\Tensor\sigmoid(tensor([$count], $input)));
    for ($i = 0; $i < $count; $i++) {
        $tails = $tails && close($got[$i], reference($input[$i]));
    }
}
var_dump($tails);

$rank4 = tensor([2, 1, 3, 2], [-2.0, -1.0, 0.5, 1.0, 2.0, 3.0, -3.0, 0.0, 0.25, 0.75, 1.5, 2.5]);
$flat = values($rank4);
$got4 = values(Toy\Tensor\sigmoid($rank4));
$rank4_ok = true;
foreach ($flat as $i => $value) {
    $rank4_ok = $rank4_ok && close($got4[$i], reference($value));
}
var_dump($rank4_ok);
var_dump(Toy\Tensor\sigmoid($rank4)->shape() === [2, 1, 3, 2]);

$sigmoid = new ReflectionFunction('Toy\\Tensor\\sigmoid');
var_dump($sigmoid->getNumberOfParameters() === 2);
var_dump($sigmoid->getNumberOfRequiredParameters() === 1);
var_dump($sigmoid->getParameters()[1]->getDefaultValue() === null);

Toy\Tensor\kernel_stats(true);
$kernel_values = values(Toy\Tensor\sigmoid($single));
$stats = Toy\Tensor\kernel_stats();
$avx2 = $stats['avx2_available'];
var_dump($stats['flf_calls'] === 1);
var_dump($stats['zif_calls'] === 0);
var_dump($avx2 ? $stats['kernel_calls'] === 1 : $stats['scalar_calls'] === 1);
var_dump($avx2 ? $stats['scalar_calls'] === 0 : true);

Toy\Tensor\kernel_enable(false);
$scalar_values = values(Toy\Tensor\sigmoid($single));
Toy\Tensor\kernel_enable(true);
$same = true;
foreach ($kernel_values as $i => $value) {
    $same = $same && close($value, $scalar_values[$i]);
}
var_dump($same);

$dynamic = 'Toy\\Tensor\\sigmoid';
var_dump($dynamic($single)->toBytes() === $result->toBytes());
var_dump(Toy\Tensor\kernel_stats()['zif_calls'] === 1);
echo "done\n";
?>
--EXPECT--
f32 [11]
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
