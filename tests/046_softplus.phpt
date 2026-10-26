--TEST--
softplus matches the reference for both branches and reuses an out tensor
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
    return $value > 20.0 ? $value : log1p(exp($value));
}
function close(float $got, float $want): bool {
    return abs($got - $want) <= 1e-6 * max(1.0, abs($want));
}

/* -100 is where the exponential underflows, 30 where the branch takes over */
$inputs = [-100.0, -20.0, -8.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 8.0, 19.0, 20.0, 30.0, 100.0];
$single = tensor([count($inputs)], $inputs);
$result = Toy\Tensor\softplus($single);
echo $result->dtype, ' ', json_encode($result->shape()), "\n";

$all_close = true;
foreach (values($result) as $i => $value) {
    $all_close = $all_close && close($value, reference($inputs[$i]));
}
var_dump($all_close);

/* Above the threshold softplus is the identity in f32 */
$large = values($result);
var_dump($large[count($inputs) - 1] === 100.0);
var_dump($large[count($inputs) - 2] === 30.0);

$out = Toy\Tensor::zeros('f32', [count($inputs)]);
var_dump(Toy\Tensor\softplus($single, $out) === $out);
var_dump($out->toBytes() === $result->toBytes());

$in_place = tensor([count($inputs)], $inputs);
var_dump(Toy\Tensor\softplus($in_place, $in_place) === $in_place);
var_dump($in_place->toBytes() === $result->toBytes());
var_dump(Toy\Tensor\softplus($single, null)->toBytes() === $result->toBytes());

/* Tails: the same loop split as sigmoid */
$tails = true;
for ($count = 1; $count <= 17; $count++) {
    $input = [];
    for ($i = 0; $i < $count; $i++) {
        $input[] = ($i % 9) - 4.0;
    }
    $got = values(Toy\Tensor\softplus(tensor([$count], $input)));
    for ($i = 0; $i < $count; $i++) {
        $tails = $tails && close($got[$i], reference($input[$i]));
    }
}
var_dump($tails);

$softplus = new ReflectionFunction('Toy\\Tensor\\softplus');
var_dump($softplus->getNumberOfParameters() === 2);
var_dump($softplus->getNumberOfRequiredParameters() === 1);

Toy\Tensor\kernel_stats(true);
$kernel_values = values(Toy\Tensor\softplus($single));
$stats = Toy\Tensor\kernel_stats();
$avx2 = $stats['avx2_available'];
var_dump($stats['flf_calls'] === 1);
var_dump($avx2 ? $stats['kernel_calls'] === 1 : $stats['scalar_calls'] === 1);
var_dump($avx2 ? $stats['scalar_calls'] === 0 : true);

Toy\Tensor\kernel_enable(false);
$scalar_values = values(Toy\Tensor\softplus($single));
Toy\Tensor\kernel_enable(true);
$same = true;
foreach ($kernel_values as $i => $value) {
    $same = $same && close($value, $scalar_values[$i]);
}
var_dump($same);
echo "done\n";
?>
--EXPECT--
f32 [13]
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
