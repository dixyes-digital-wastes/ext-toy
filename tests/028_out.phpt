--TEST--
Optional out writes into existing tensors and returns them
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function bytes(Toy\Tensor $tensor): string {
    return $tensor->toBytes();
}

$a = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$b = tensor([4], [5.0, 6.0, 7.0, 8.0]);
$expected = Toy\Tensor\add($a, $b);
$out = Toy\Tensor::zeros('f32', [4]);
$result = Toy\Tensor\add($a, $b, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));
var_dump(bytes($a) === pack('g*', 1.0, 2.0, 3.0, 4.0));

$left = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$expected = Toy\Tensor\add($left, $b);
$result = Toy\Tensor\add($left, $b, $left);
var_dump($result === $left);
var_dump(bytes($left) === bytes($expected));

$right = tensor([4], [5.0, 6.0, 7.0, 8.0]);
$expected = Toy\Tensor\add($a, $right);
$result = Toy\Tensor\add($a, $right, $right);
var_dump($result === $right);
var_dump(bytes($right) === bytes($expected));

$source = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$expected = Toy\Tensor\scale($source, 2.0);
$out = Toy\Tensor::zeros('f32', [4]);
$result = Toy\Tensor\scale($source, 2.0, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));
$result = Toy\Tensor\scale($source, 3.0, $source);
var_dump($result === $source);
var_dump(bytes($source) === pack('g*', 3.0, 6.0, 9.0, 12.0));

$a2 = tensor([2, 2], [1.0, 2.0, 3.0, 4.0]);
$b2 = tensor([2, 2], [5.0, 6.0, 7.0, 8.0]);
$expected = Toy\Tensor\matmul_f32($a2, $b2, 2, 2, 2);
$out = Toy\Tensor::zeros('f32', [2, 2]);
$result = Toy\Tensor\matmul_f32($a2, $b2, 2, 2, 2, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));

$x = tensor([2], [1.0, 2.0]);
$expected = Toy\Tensor\matvec_f32($a2, $x, 2, 2);
$out = Toy\Tensor::zeros('f32', [2]);
$result = Toy\Tensor\matvec_f32($a2, $x, 2, 2, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));

$q8 = tensor([2, 32], array_fill(0, 64, 1.0))->cast('q8_0');
$qx = tensor([32, 1], array_fill(0, 32, 1.0));
$expected = Toy\Tensor\matmul_q8_0_f32($q8, $qx, 2, 1, 32);
$out = Toy\Tensor::zeros('f32', [2, 1]);
$result = Toy\Tensor\matmul_q8_0_f32($q8, $qx, 2, 1, 32, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));

$qv = tensor([32], array_fill(0, 32, 1.0));
$expected = Toy\Tensor\matvec_q8_0_f32($q8, $qv, 2, 32);
$out = Toy\Tensor::zeros('f32', [2]);
$result = Toy\Tensor\matvec_q8_0_f32($q8, $qv, 2, 32, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));

$norm_x = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$norm_w = tensor([4], [1.0, 0.5, 2.0, 1.5]);
$expected = Toy\Tensor\rmsnorm($norm_x, $norm_w);
$inplace = tensor([4], [1.0, 2.0, 3.0, 4.0]);
$result = Toy\Tensor\rmsnorm_into($inplace, $norm_w, $inplace);
var_dump($result === $inplace);
var_dump(bytes($inplace) === bytes($expected));

$out = Toy\Tensor::zeros('f32', [4]);
$result = Toy\Tensor\rmsnorm_into($norm_x, $norm_w, $out);
var_dump($result === $out);
var_dump(bytes($out) === bytes($expected));

$expected = Toy\Tensor\add($a, $b);
$result = Toy\Tensor\add($a, $b, null);
var_dump($result !== $a);
var_dump(bytes($result) === bytes($expected));
var_dump($out->shape() === [4]);
var_dump($out->ndim() === 1);
var_dump($out->bytes() === 16);
var_dump($inplace->shape() === [4]);
var_dump($result instanceof Toy\Tensor);
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
