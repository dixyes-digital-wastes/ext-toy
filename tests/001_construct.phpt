--TEST--
Toy\Tensor construction, metadata, reshape, and cast
--EXTENSIONS--
toy
--FILE--
<?php
function floats_to_bytes(array $values): string { return pack('g*', ...$values); }
function bytes_to_floats(string $bytes): array { return array_values(unpack('g*', $bytes)); }

$property = new ReflectionProperty(Toy\Tensor::class, 'dtype');
var_dump($property->isReadOnly());

$zeros = Toy\Tensor::zeros('f32', [2, 3]);
var_dump($zeros->dtype);
var_dump($zeros->shape());
var_dump($zeros->ndim());
var_dump($zeros->bytes());
var_dump($zeros->toBytes() === str_repeat("\0", 24));

$ones = Toy\Tensor::ones('f32', [2]);
var_dump(bytes_to_floats($ones->toBytes()));

$full = Toy\Tensor::full('f32', [2], 0.5);
var_dump(bytes_to_floats($full->toBytes()));

$source = Toy\Tensor::fromBytes('f32', [2, 2], floats_to_bytes([1.0, 2.0, 3.0, 4.0]));
var_dump($source->shape());
var_dump($source->toBytes() === floats_to_bytes([1.0, 2.0, 3.0, 4.0]));
$same = $source->reshape([4]);
var_dump($same === $source);
var_dump($source->shape());
var_dump($source->toBytes() === floats_to_bytes([1.0, 2.0, 3.0, 4.0]));
$cast = $source->cast('f32');
var_dump($cast !== $source);
var_dump($cast->shape());
var_dump($cast->toBytes() === $source->toBytes());
var_dump(function_exists('Toy\\Tensor\\alias'));
var_dump(method_exists(Toy\Tensor::class, 'alias'));
echo "done\n";
?>
--EXPECT--
bool(true)
string(3) "f32"
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(3)
}
int(2)
int(24)
bool(true)
array(2) {
  [0]=>
  float(1)
  [1]=>
  float(1)
}
array(2) {
  [0]=>
  float(0.5)
  [1]=>
  float(0.5)
}
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
bool(true)
bool(true)
array(1) {
  [0]=>
  int(4)
}
bool(true)
bool(true)
array(1) {
  [0]=>
  int(4)
}
bool(true)
bool(false)
bool(false)
done
