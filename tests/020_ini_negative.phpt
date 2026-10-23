--TEST--
A negative kernel_arena_size disables the arena silently
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=-1
--FILE--
<?php
ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena => disabled'));

$value = Toy\Tensor\scale(Toy\Tensor::full('f32', [3], 2.0), 4.0);
var_dump(array_values(unpack('g*', $value->toBytes())) === [8.0, 8.0, 8.0]);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 0);
var_dump($stats['scalar_calls'] > 0);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
done
