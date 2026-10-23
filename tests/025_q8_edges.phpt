--TEST--
q8_0 edge blocks, f16 scales and larger kernel tiles
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

function block(int $scale_bits, int $value, int $count = 32): string {
    return pack('v', $scale_bits) . str_repeat(chr($value & 0xFF), $count);
}

var_dump(Toy\Tensor::zeros('q8_0', [32])->cast('f32')->toBytes() === str_repeat("\0", 128));

$subnormal = Toy\Tensor::fromBytes('q8_0', [32], block(0x0001, 1))->cast('f32');
$floats = test_floats($subnormal);
var_dump($floats[0] > 0.0 && $floats[0] < 1e-6);
$negative = Toy\Tensor::fromBytes('q8_0', [32], block(0x8001, 1))->cast('f32');
var_dump(test_floats($negative)[0] < 0.0);

$infinite = Toy\Tensor::fromBytes('q8_0', [32], block(0x7C00, 1))->cast('f32');
var_dump(is_infinite(test_floats($infinite)[0]));
$nan = Toy\Tensor::fromBytes('q8_0', [32], block(0x7E00, 1))->cast('f32');
var_dump(is_nan(test_floats($nan)[0]));

$tiny = Toy\Tensor::full('f32', [32], 1e-30)->cast('q8_0');
var_dump(unpack('v', substr($tiny->toBytes(), 0, 2))[1] === 0);
var_dump($tiny->cast('f32')->toBytes() === str_repeat("\0", 128));

$small = Toy\Tensor::full('f32', [32], 1e-4)->cast('q8_0');
$round = test_floats($small->cast('f32'));
var_dump(abs($round[0] - 1e-4) <= 2e-6);

$zero = Toy\Tensor::full('q8_0', [32], 0.0);
var_dump($zero->cast('f32')->toBytes() === str_repeat("\0", 128));

var_dump(Toy\Tensor::zeros('q8_0', [4, 96])->bytes());

$m = 1;
$n = 1;
$k = 32;
$weight = Toy\Tensor::full('f32', [$m, $k], 1.0)->cast('q8_0');
$vector = Toy\Tensor::full('f32', [$k], 2.0);
$product = Toy\Tensor\matvec_q8_0_f32($weight, $vector, $m, $k);
var_dump(abs(test_floats($product)[0] - 64.0) < 0.02);

$m = 33;
$n = 33;
$k = 64;
$wv = [];
for ($i = 0; $i < $m * $k; $i++) {
    $wv[] = test_value($i, 5);
}
$xv = [];
for ($i = 0; $i < $k * $n; $i++) {
    $xv[] = test_value($i, 6);
}
$w = test_tensor([$m, $k], $wv)->cast('q8_0');
$x = test_tensor([$k, $n], $xv);
$out = test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
$blocks = test_q8_blocks($w->toBytes());
[$ref, $absSums] = test_q8_matmul_ref($blocks, $xv, $m, $n, $k);
var_dump(test_q8_metrics($out, $ref, $absSums)['maxScaled'] < 4e-6);

Toy\Tensor\kernel_isa('scalar');
$scalar = test_floats(Toy\Tensor\matmul_q8_0_f32($w, $x, $m, $n, $k));
var_dump(test_q8_metrics($scalar, $ref, $absSums)['maxScaled'] < 4e-6);
Toy\Tensor\kernel_isa('avx2');
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
int(408)
bool(true)
bool(true)
bool(true)
done
