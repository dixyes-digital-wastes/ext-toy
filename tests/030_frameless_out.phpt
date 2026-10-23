--TEST--
Two-argument static add and scale calls stay frameless with a null out
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}

$a = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([4], [5.0, 6.0, 7.0, 8.0]);

$add = new ReflectionFunction('Toy\\Tensor\\add');
var_dump($add->getNumberOfParameters() === 3);
var_dump($add->getNumberOfRequiredParameters() === 2);
var_dump($add->getParameters()[2]->isOptional());
var_dump($add->getParameters()[2]->getDefaultValue() === null);

$scale = new ReflectionFunction('Toy\\Tensor\\scale');
var_dump($scale->getNumberOfParameters() === 3);
var_dump($scale->getNumberOfRequiredParameters() === 2);

try {
    Toy\Tensor\add($a, $b, 1);
} catch (Throwable $e) {
    echo get_class($e), "\n";
}
try {
    Toy\Tensor\scale($a, 2.0, []);
} catch (Throwable $e) {
    echo get_class($e), "\n";
}

Toy\Tensor\kernel_stats(true);
$expected = Toy\Tensor\add($a, $b);
$out = Toy\Tensor::zeros('f32', [4]);
$reuse = Toy\Tensor\add($a, $b, $out);
$explicit_null = Toy\Tensor\add($a, $b, null);
$scaled = Toy\Tensor\scale($a, 2.0);
$scale_out = Toy\Tensor::zeros('f32', [4]);
$scale_reuse = Toy\Tensor\scale($a, 2.0, $scale_out);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['flf_calls'] === 5);
var_dump($stats['zif_calls'] === 0);
var_dump($reuse === $out);
var_dump($reuse->toBytes() === $expected->toBytes());
var_dump($explicit_null->toBytes() === $expected->toBytes());
var_dump($scale_reuse === $scale_out);
var_dump($scaled->toBytes() === $scale_out->toBytes());

$dynamic = 'Toy\\Tensor\\add';
$dynamic_out = Toy\Tensor::zeros('f32', [4]);
$dynamic_result = $dynamic($a, $b, $dynamic_out);
var_dump($dynamic_result === $dynamic_out);
var_dump(Toy\Tensor\kernel_stats()['zif_calls'] === 1);
var_dump($dynamic_result !== $a);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
TypeError
TypeError
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
