--TEST--
An invalid kernel_arena_size falls back to the default arena
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=abc
display_startup_errors=0
--FILE--
<?php
ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena => 33554432 bytes'));

$value = Toy\Tensor\add(
    Toy\Tensor::full('f32', [4], 1.0),
    Toy\Tensor::full('f32', [4], 2.0));
var_dump(array_values(unpack('g*', $value->toBytes())));
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] > 0);
echo "done\n";
?>
--EXPECTF--
PHP Warning:  PHP Startup: toy.kernel_arena_size value %s
bool(true)
array(4) {
  [0]=>
  float(3)
  [1]=>
  float(3)
  [2]=>
  float(3)
  [3]=>
  float(3)
}
bool(true)
done
