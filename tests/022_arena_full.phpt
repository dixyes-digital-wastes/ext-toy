--TEST--
A small arena fills after four kernels and the later calls fall back
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=64K
--FILE--
<?php
$pairs = [];
$results = [];
foreach ([8, 9, 10, 11, 12] as $count) {
    $pairs[$count] = [
        Toy\Tensor::full('f32', [$count], 1.0),
        Toy\Tensor::full('f32', [$count], 2.0),
    ];
    $value = Toy\Tensor\add($pairs[$count][0], $pairs[$count][1]);
    $results[] = array_values(unpack('g*', $value->toBytes())) === array_fill(0, $count, 3.0);
}
var_dump($results);

$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated']);
var_dump($stats['generation_failures'] > 0);
var_dump($stats['scalar_calls'] > 0);

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, '16 of 16 pages used'));

$value = Toy\Tensor\add($pairs[8][0], $pairs[8][1]);
var_dump(array_values(unpack('g*', $value->toBytes())) === array_fill(0, 8, 3.0));
var_dump(Toy\Tensor\kernel_stats()['memo_hits'] > 0);
echo "done\n";
?>
--EXPECT--
array(5) {
  [0]=>
  bool(true)
  [1]=>
  bool(true)
  [2]=>
  bool(true)
  [3]=>
  bool(true)
  [4]=>
  bool(true)
}
int(4)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
