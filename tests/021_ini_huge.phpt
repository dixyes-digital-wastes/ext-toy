--TEST--
An arena larger than the tier limit is rejected and disables the arena
--EXTENSIONS--
toy
--INI--
toy.kernel_arena_size=5G
display_startup_errors=0
--FILE--
<?php
ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena => disabled'));

$value = Toy\Tensor\add(
    Toy\Tensor::full('f32', [2], -1.0),
    Toy\Tensor::full('f32', [2], 3.0));
var_dump(array_values(unpack('g*', $value->toBytes())) === [2.0, 2.0]);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] === 0);
var_dump($stats['scalar_calls'] > 0);
echo "done\n";
?>
--EXPECTF--
PHP Warning:  PHP Startup: cannot map %s
bool(true)
bool(true)
bool(true)
bool(true)
done
