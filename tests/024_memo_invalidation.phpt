--TEST--
Tensor memo survives reshape and is skipped by new tensors and new parameters
--EXTENSIONS--
toy
--FILE--
<?php
function values(Toy\Tensor $tensor): array {
    return array_values(unpack('g*', $tensor->toBytes()));
}

$a = Toy\Tensor::full('f32', [4], 1.0);
$b = Toy\Tensor::full('f32', [4], 2.0);

Toy\Tensor\kernel_stats(true);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['memo_misses'] === 1);
var_dump($stats['memo_hits'] === 1);

$a->reshape([2, 2]);
$b->reshape([2, 2]);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
var_dump(Toy\Tensor\kernel_stats()['memo_hits'] === 2);

$scaled = Toy\Tensor\scale($a, 2.0);
var_dump(values($scaled) === [2.0, 2.0, 2.0, 2.0]);
var_dump(values(Toy\Tensor\scale($a, 2.0)) === [2.0, 2.0, 2.0, 2.0]);
var_dump(values(Toy\Tensor\scale($a, 3.0)) === [3.0, 3.0, 3.0, 3.0]);
var_dump(values(Toy\Tensor\scale($a, 3.0)) === [3.0, 3.0, 3.0, 3.0]);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['memo_misses'] === 3);
var_dump($stats['memo_hits'] === 4);

$copy = $a->cast('f32');
var_dump(values(Toy\Tensor\add($copy, $b)) === [3.0, 3.0, 3.0, 3.0]);
var_dump(Toy\Tensor\kernel_stats()['memo_misses'] === 4);
var_dump(values(Toy\Tensor\add($copy, $b)) === [3.0, 3.0, 3.0, 3.0]);

Toy\Tensor\kernel_enable(false);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
Toy\Tensor\kernel_enable(true);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
var_dump(values(Toy\Tensor\add($a, $b)) === [3.0, 3.0, 3.0, 3.0]);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['memo_hits'] === 6);
var_dump($stats['memo_misses'] === 5);
var_dump($stats['scalar_calls'] === 1);
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
bool(true)
bool(true)
done
