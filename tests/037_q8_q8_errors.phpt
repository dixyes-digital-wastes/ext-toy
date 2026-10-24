--TEST--
q8_0 x q8_0 ops reject wrong dtypes, shapes, byte counts and out tensors
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
        echo get_class($e), " ", $e->getMessage(), "\n";
    }
}

$block = ['d' => 0x2C00, 'q' => array_fill(0, 32, 1)];
$w = Toy\Tensor::fromBytes('q8_0', [2, 64], test_q8_pack(array_fill(0, 4, $block)));
$x = Toy\Tensor::fromBytes('q8_0', [64], test_q8_pack(array_fill(0, 2, $block)));
$wide = Toy\Tensor::fromBytes('q8_0', [64, 32], test_q8_pack(array_fill(0, 64, $block)));
$f32 = test_tensor([2, 64], array_fill(0, 128, 1.0));

// dtype of both operands
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($f32, $x, 2, 64));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $f32, 2, 64));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $f32, 2, 32, 64));

// k must be a multiple of the 32 value block
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 33));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 32, 63));

// shapes must agree with the arguments
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 3, 64));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 128));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $wide, 2, 64));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $x, 2, 32, 64));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 64, 64));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 32, 32));

// m, n and k must be positive
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 0, 64));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, -1, 64));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 0));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 0, 64));

// out must be an f32 tensor of the result shape
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 64, 1));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 64, $w));
show(fn() => Toy\Tensor\matvec_q8_0_q8_0($w, $x, 2, 64, test_tensor([3], [1.0, 2.0, 3.0])));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 32, 64, test_tensor([2, 31], array_fill(0, 62, 1.0))));
show(fn() => Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 32, 64, Toy\Tensor::zeros('q8_0', [64, 32])));

/*
 * The activation holds one q8_0 vector of k values per column, so the q8_0
 * dtype rule puts the block axis on the last dimension and a [k, n] activation
 * needs n to be a multiple of 32
 */
show(fn() => Toy\Tensor::zeros('q8_0', [64, 33]));
show(fn() => Toy\Tensor::zeros('q8_0', [33]));
show(fn() => Toy\Tensor::fromBytes('q8_0', [64, 32], str_repeat("\0", 34)));

// A well formed call still works after all the rejections
$product = Toy\Tensor\matmul_q8_0_q8_0($w, $wide, 2, 32, 64);
var_dump($product->shape() === [2, 32]);
var_dump(count(array_values(unpack('g*', $product->toBytes()))) === 64);
echo "done\n";
?>
--EXPECT--
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #1 ($w) must be a q8_0 tensor
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #2 ($x) must be a q8_0 tensor
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #2 ($x) must be a q8_0 tensor
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #4 ($k) must be a multiple of 32
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #5 ($k) must be a multiple of 32
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #1 ($w) must have shape [3, 64], got [2, 64]
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #1 ($w) must have shape [2, 128], got [2, 64]
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #2 ($x) must be a 1-D tensor
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #2 ($x) must be a 2-D tensor
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #2 ($x) must have shape [64, 64], got [64, 32]
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #1 ($w) must have shape [2, 32], got [2, 64]
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #3 ($m) must be between 1 and 4294967295
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #3 ($m) must be between 1 and 4294967295
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #4 ($k) must be between 1 and 4294967295
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #4 ($n) must be between 1 and 4294967295
TypeError Toy\Tensor\matvec_q8_0_q8_0(): Argument #5 ($out) must be of type ?Toy\Tensor, int given
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #5 ($out) must be an f32 tensor
ValueError Toy\Tensor\matvec_q8_0_q8_0(): Argument #5 ($out) must have the same shape as the result
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #6 ($out) must have the same shape as the result
ValueError Toy\Tensor\matmul_q8_0_q8_0(): Argument #6 ($out) must be an f32 tensor
ValueError Toy\Tensor::zeros(): Argument #2 ($shape) is invalid
ValueError Toy\Tensor::zeros(): Argument #2 ($shape) is invalid
ValueError Toy\Tensor::fromBytes(): Argument #3 ($bytes) must be exactly 2176 bytes, 34 given
bool(true)
bool(true)
done
