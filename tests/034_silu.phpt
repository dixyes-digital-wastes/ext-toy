--TEST--
silu applies the scalar reference and reuses an out tensor
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
function reference(float $x): float {
    return $x * (1 / (1 + exp(-$x)));
}
function show_floats(Toy\Tensor $tensor): void {
    echo json_encode(array_map(fn(float $v): float => round($v, 6), values($tensor))), "\n";
}

$input = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
$a = tensor([count($input)], $input);
$result = Toy\Tensor\silu($a);
echo $result->dtype, ' ', json_encode($result->shape()), "\n";
show_floats($result);

$close = true;
foreach (values($result) as $i => $value) {
    $close = $close && abs($value - reference($input[$i])) < 1e-6;
}
var_dump($close);

$out = Toy\Tensor::zeros('f32', [count($input)]);
var_dump(Toy\Tensor\silu($a, $out) === $out);
var_dump($out->toBytes() === $result->toBytes());
var_dump(Toy\Tensor\silu($a, null)->toBytes() === $result->toBytes());

$in_place = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$expected = Toy\Tensor\silu($in_place);
var_dump(Toy\Tensor\silu($in_place, $in_place) === $in_place);
var_dump($in_place->toBytes() === $expected->toBytes());

Toy\Tensor\kernel_stats(true);
Toy\Tensor\silu($a);
$default_out = Toy\Tensor\silu($a, $out);
$stats = Toy\Tensor\kernel_stats();
var_dump($default_out === $out);
var_dump($stats['flf_calls'] === 2);
var_dump($stats['zif_calls'] === 0);
var_dump($stats['scalar_calls'] >= 2);
var_dump($stats['generated'] === 0);

$dynamic = 'Toy\\Tensor\\silu';
$dynamic($a);
var_dump(Toy\Tensor\kernel_stats()['zif_calls'] === 1);

$silu = new ReflectionFunction('Toy\\Tensor\\silu');
var_dump($silu->getNumberOfParameters() === 2);
var_dump($silu->getNumberOfRequiredParameters() === 1);
var_dump($silu->getParameters()[1]->isOptional());
var_dump($silu->getParameters()[1]->getDefaultValue() === null);

$big = tensor([1024], array_map(fn(int $i): float => ($i % 11) - 5, range(0, 1023)));
$kernel_enabled = Toy\Tensor\silu($big);
Toy\Tensor\kernel_enable(false);
$scalar = Toy\Tensor\silu($big);
Toy\Tensor\kernel_enable(true);
var_dump($kernel_enabled->toBytes() === $scalar->toBytes());
echo "done\n";
?>
--EXPECT--
f32 [7]
[-0.142278,-0.238406,-0.268941,0,0.731059,1.761594,2.857722]
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
