--TEST--
quantize_q8_0 rejects wrong dtypes, shapes, lengths and row ranges
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
        echo get_class($e), ': ', $e->getMessage(), "\n";
    }
}

$twos = test_tensor([2, 64], array_fill(0, 128, 1.0));
$vector = test_tensor([64], array_fill(0, 64, 1.0));
$ragged = test_tensor([2, 33], array_fill(0, 66, 1.0));
$cube = test_tensor([2, 2, 32], array_fill(0, 128, 1.0));

show(fn () => Toy\Tensor\quantize_q8_0($twos->cast('q8_0')));
show(fn () => Toy\Tensor\quantize_q8_0($ragged));
show(fn () => Toy\Tensor\quantize_q8_0($cube));
show(fn () => Toy\Tensor\quantize_q8_0($twos, -1));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 3));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 1, 1));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 1, 0));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 1, 3));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 2));
show(fn () => Toy\Tensor\quantize_q8_0($vector, 1));
show(fn () => Toy\Tensor\quantize_q8_0($vector, 0, 2));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 'a'));
show(fn () => Toy\Tensor\quantize_q8_0(null));

// A numeric string is coerced, the same as the m and k arguments of the other ops
show(fn () => Toy\Tensor\quantize_q8_0($twos, '0', '1'));

// The valid calls stay valid
show(fn () => Toy\Tensor\quantize_q8_0($twos));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 1));
show(fn () => Toy\Tensor\quantize_q8_0($twos, 0, 2));
show(fn () => Toy\Tensor\quantize_q8_0($vector, 0, 1));
?>
--EXPECT--
ValueError: Toy\Tensor\quantize_q8_0(): Argument #1 ($x) must be an f32 tensor
ValueError: Toy\Tensor\quantize_q8_0(): Argument #1 ($x) must have a last dimension that is a multiple of 32
ValueError: Toy\Tensor\quantize_q8_0(): Argument #1 ($x) must be a 1-D or 2-D tensor
ValueError: Toy\Tensor\quantize_q8_0(): Argument #2 ($rowBegin) must be between 0 and 1
ValueError: Toy\Tensor\quantize_q8_0(): Argument #2 ($rowBegin) must be between 0 and 1
ValueError: Toy\Tensor\quantize_q8_0(): Argument #3 ($rowEnd) must be greater than argument #2 ($rowBegin)
ValueError: Toy\Tensor\quantize_q8_0(): Argument #3 ($rowEnd) must be greater than argument #2 ($rowBegin)
ValueError: Toy\Tensor\quantize_q8_0(): Argument #3 ($rowEnd) must be between 0 and 2
ValueError: Toy\Tensor\quantize_q8_0(): Argument #2 ($rowBegin) must be between 0 and 1
ValueError: Toy\Tensor\quantize_q8_0(): Argument #2 ($rowBegin) must be between 0 and 0
ValueError: Toy\Tensor\quantize_q8_0(): Argument #3 ($rowEnd) must be between 0 and 1
TypeError: Toy\Tensor\quantize_q8_0(): Argument #2 ($rowBegin) must be of type int, string given
TypeError: Toy\Tensor\quantize_q8_0(): Argument #1 ($x) must be of type Toy\Tensor, null given
no exception
no exception
no exception
no exception
no exception
