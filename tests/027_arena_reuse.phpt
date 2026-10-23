--TEST--
A freed arena block is reused after a generation failure
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=64K
--FILE--
<?php
$pairs = [];
foreach ([8, 9] as $count) {
    $pairs[$count] = [
        Toy\Tensor::full('f32', [$count], 1.0),
        Toy\Tensor::full('f32', [$count], 2.0),
    ];
    Toy\Tensor\add($pairs[$count][0], $pairs[$count][1]);
}
var_dump(Toy\Tensor\kernel_stats()['generated'] === 2);

$m = 65536;
$k = 32;
$weight = Toy\Tensor::full('q8_0', [$m, $k], 1.0);
$vector = Toy\Tensor::full('f32', [$k], 2.0);
Toy\Tensor\matvec_q8_0_f32($weight, $vector, $m, $k);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generation_failures'] === 1);
var_dump($stats['scalar_calls'] === 1);

foreach ([10, 11] as $count) {
    $pairs[$count] = [
        Toy\Tensor::full('f32', [$count], 1.0),
        Toy\Tensor::full('f32', [$count], 2.0),
    ];
    Toy\Tensor\add($pairs[$count][0], $pairs[$count][1]);
}
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 4);

$value = Toy\Tensor\add(
    Toy\Tensor::full('f32', [12], 1.0),
    Toy\Tensor::full('f32', [12], 2.0));
var_dump(array_values(unpack('g*', $value->toBytes())) === array_fill(0, 12, 3.0));
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generation_failures'] === 2);
var_dump($stats['scalar_calls'] === 2);

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, '16 of 16 pages used'));

$again = Toy\Tensor\add($pairs[8][0], $pairs[8][1]);
var_dump(array_values(unpack('g*', $again->toBytes())) === array_fill(0, 8, 3.0));
var_dump(Toy\Tensor\kernel_stats()['memo_hits'] > 0);
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
