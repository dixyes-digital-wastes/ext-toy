--TEST--
mul computes the element-wise product and reuses an out tensor
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function show_floats(Toy\Tensor $tensor): void {
    $values = array_values(unpack('g*', $tensor->toBytes()));
    echo json_encode(array_map(fn(float $v): float => round($v, 6), $values)), "\n";
}

$a = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([4], [0.5, -2.0, 3.0, 0.0]);

$product = Toy\Tensor\mul($a, $b);
echo $product->dtype, ' ', json_encode($product->shape()), "\n";
show_floats($product);
var_dump($product->toBytes() === tensor([4], [0.5, -4.0, 9.0, 0.0])->toBytes());
var_dump($product->toBytes() === Toy\Tensor\mul($a, $b, null)->toBytes());

$out = Toy\Tensor::zeros('f32', [4]);
$reuse = Toy\Tensor\mul($a, $b, $out);
var_dump($reuse === $out);
var_dump($out->toBytes() === $product->toBytes());

$left = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$right = tensor([2, 2], [2.0, 3.0, 4.0, 5.0]);
$reference = Toy\Tensor\mul($left, $right);
var_dump(Toy\Tensor\mul($left, $right, $left) === $left);
var_dump($left->toBytes() === $reference->toBytes());

$left = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$right = tensor([2, 2], [2.0, 3.0, 4.0, 5.0]);
$reference = Toy\Tensor\mul($left, $right);
var_dump(Toy\Tensor\mul($left, $right, $right) === $right);
var_dump($right->toBytes() === $reference->toBytes());

$square = tensor([3], [-2.0, 3.0, 4.0]);
var_dump(Toy\Tensor\mul($square, $square, $square) === $square);
show_floats($square);

Toy\Tensor\kernel_stats(true);
Toy\Tensor\mul($a, $b);
$stats = Toy\Tensor\kernel_stats();
var_dump($stats['flf_calls'] === 1);
var_dump($stats['zif_calls'] === 0);
var_dump($stats['memo_hits'] === 1);

$dynamic = 'Toy\\Tensor\\mul';
$dynamic($a, $b);
var_dump(Toy\Tensor\kernel_stats()['zif_calls'] === 1);

$kernel_ok = true;
$manual_ok = true;
foreach ([1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 64, 1024] as $count) {
    $left = tensor([$count], array_map(fn(int $i): float => $i + 1, range(0, $count - 1)));
    $right = tensor([$count], array_map(fn(int $i): float => $i % 3 - 1, range(0, $count - 1)));
    Toy\Tensor\kernel_enable(true);
    $kernel = Toy\Tensor\mul($left, $right);
    Toy\Tensor\kernel_enable(false);
    $scalar = Toy\Tensor\mul($left, $right);
    Toy\Tensor\kernel_enable(true);
    $kernel_ok = $kernel_ok && $kernel->toBytes() === $scalar->toBytes();
    $manual = array_map(fn(int $i): float => ($i + 1) * ($i % 3 - 1), range(0, $count - 1));
    $manual_ok = $manual_ok && $kernel->toBytes() === tensor([$count], $manual)->toBytes();
}
var_dump($kernel_ok);
var_dump($manual_ok);
echo "done\n";
?>
--EXPECT--
f32 [4]
[0.5,-4,9,0]
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
[4,9,16]
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
