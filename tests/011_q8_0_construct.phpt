--TEST--
q8_0 construction, byte layout and reshape constraints
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

$zeros = Toy\Tensor::zeros('q8_0', [2, 32]);
var_dump($zeros->dtype);
var_dump($zeros->shape());
var_dump($zeros->ndim());
var_dump($zeros->bytes());
var_dump($zeros->toBytes() === str_repeat("\0", 68));

$ones = Toy\Tensor::ones('q8_0', [32]);
var_dump($ones->toBytes() === "\x08\x20" . str_repeat("\x7f", 32));

$full = Toy\Tensor::full('q8_0', [64], -1.0);
var_dump($full->bytes());
var_dump($full->toBytes()
    === "\x08\x20" . str_repeat("\x81", 32) . "\x08\x20" . str_repeat("\x81", 32));

$block = "\x00\x3c" . str_repeat("\x7f", 32);
$from = Toy\Tensor::fromBytes('q8_0', [32], $block);
var_dump($from->toBytes() === $block);
var_dump($from->cast('f32')->toBytes() === pack('g*', ...array_fill(0, 32, 127.0)));

$wide = Toy\Tensor::zeros('q8_0', [2, 32]);
$same = $wide->reshape([64]);
var_dump($same === $wide);
var_dump($wide->shape());
var_dump(strlen($wide->toBytes()));

show(fn() => Toy\Tensor::zeros('q8_0', [33]));
show(fn() => Toy\Tensor::zeros('q8_0', [2, 33]));
show(fn() => Toy\Tensor::zeros('q8_0', [0, 32]));
show(fn() => Toy\Tensor::fromBytes('q8_0', [32], str_repeat("\0", 33)));
show(fn() => Toy\Tensor::fromBytes('q8_0', [32], str_repeat("\0", 68)));
show(fn() => $wide->reshape([65]));
show(fn() => Toy\Tensor::zeros('f16', [32]));
show(fn() => Toy\Tensor::full('q8_0', [32], 1.0)->reshape([2, 16]));
echo "done\n";
?>
--EXPECT--
string(4) "q8_0"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(32)
}
int(2)
int(68)
bool(true)
bool(true)
int(68)
bool(true)
bool(true)
bool(true)
bool(true)
array(1) {
  [0]=>
  int(64)
}
int(68)
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
done
