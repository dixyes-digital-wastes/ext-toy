--TEST--
A spec that the generator rejects frees its block and falls back
--EXTENSIONS--
toy
--FILE--
<?php
$m = 70000;
$k = 1;
$n = 1;

$w = Toy\Tensor::full('f32', [$m, $k], 2.0);
$x = Toy\Tensor::full('f32', [$k, $n], 3.0);
$value = Toy\Tensor\matmul_f32($w, $x, $m, $n, $k);
$floats = unpack('g*', $value->toBytes());
var_dump($floats[1] === 6.0);
var_dump($floats[$m] === 6.0);

$vector = Toy\Tensor::full('f32', [$k], 3.0);
$product = Toy\Tensor\matvec_f32($w, $vector, $m, $k);
$floats = unpack('g*', $product->toBytes());
var_dump($floats[1] === 6.0);
var_dump($floats[$m] === 6.0);

$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 0);
var_dump($stats['generation_failures'] === 2);
var_dump($stats['scalar_calls'] === 2);

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, '0 of 8192 pages used'));

$small = Toy\Tensor\matmul_f32(
    Toy\Tensor::full('f32', [2, 2], 1.0),
    Toy\Tensor::full('f32', [2, 2], 2.0),
    2, 2, 2);
var_dump(array_values(unpack('g*', $small->toBytes())) === [4.0, 4.0, 4.0, 4.0]);
var_dump(Toy\Tensor\kernel_stats()['generated'] === 1);
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
done
