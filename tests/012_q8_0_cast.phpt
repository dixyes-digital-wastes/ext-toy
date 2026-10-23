--TEST--
q8_0 cast follows the ggml quantization rule and round-trips
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}

$values = [];
for ($i = 0; $i < 64; $i++) {
    $values[] = toy_test_value($i, 15) * (1.0 + ($i % 5) * 0.8);
}

$source = toy_test_tensor([2, 32], $values);
$quantized = $source->cast('q8_0');
var_dump($quantized->dtype);
var_dump($quantized->bytes());
var_dump($source->dtype);
var_dump($source->bytes());

$blocks = toy_test_q8_blocks($quantized->toBytes());
var_dump(count($blocks));

$layoutOk = true;
$maxStep = 0.0;
for ($b = 0; $b < 2; $b++) {
    $chunk = array_slice($values, $b * 32, 32);
    $amax = 0.0;
    foreach ($chunk as $value) {
        $amax = max($amax, abs($value));
    }
    $d = toy_test_f32($amax / 127.0);
    $id = toy_test_f32(1.0 / $d);
    $maxStep = max($maxStep, toy_test_f16_to_f32(toy_test_f32_to_f16_bits($d)));

    $layoutOk = $layoutOk
        && toy_test_f32_to_f16_bits($d) === unpack('v', substr($quantized->toBytes(), $b * 34, 2))[1];
    for ($i = 0; $i < 32; $i++) {
        $scaled = toy_test_f32($chunk[$i] * $id);
        $q = (int) round($scaled);
        $q = max(-127, min(127, $q));
        $layoutOk = $layoutOk && $q === $blocks[$b]['q'][$i];
    }
}
var_dump($layoutOk);

$dequantized = $quantized->cast('f32');
$expected = [];
for ($b = 0; $b < 2; $b++) {
    foreach ($blocks[$b]['q'] as $q) {
        $expected[] = $q * $blocks[$b]['d'];
    }
}
var_dump($dequantized->toBytes() === pack('g*', ...$expected));

$maxError = 0.0;
foreach ($values as $i => $value) {
    $maxError = max($maxError, abs($expected[$i] - $value));
}
var_dump($maxError <= $maxStep / 2.0 + 1e-6);

$again = $quantized->cast('q8_0');
var_dump($again !== $quantized);
var_dump($again->toBytes() === $quantized->toBytes());

$copy = $source->cast('f32');
var_dump($copy !== $source);
var_dump($copy->toBytes() === $source->toBytes());

show(fn() => toy_test_tensor([33], array_fill(0, 33, 1.0))->cast('q8_0'));
show(fn() => $source->cast('f16'));
show(fn() => $quantized->cast('f16'));
show(fn() => Toy\Tensor::zeros('f32', [3, 33])->cast('q8_0'));
echo "done\n";
?>
--EXPECT--
string(4) "q8_0"
int(68)
string(3) "f32"
int(256)
int(2)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
ValueError
ValueError
ValueError
ValueError
done
