--TEST--
shift_rows moves the history rows up and can append a new last row
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function values(Toy\Tensor $tensor): array {
    return array_values(unpack('g*', $tensor->toBytes()));
}

/* Four rows of three channels, the newest step last */
$rows = 3;
$width = 3;
$state = tensor([4, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
$new = tensor([3], [13.0, 14.0, 15.0]);

$shifted = Toy\Tensor\shift_rows($state, $new);
var_dump($shifted->shape() === [4, 3]);
var_dump(values($shifted) === [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
var_dump(values($state) === [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

/* Without a row the last row keeps its value */
var_dump(values(Toy\Tensor\shift_rows($state)) ===
    [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0]);

/* In place: one call shifts and appends */
$in_place = tensor([4, 3], range(1, 12));
var_dump(Toy\Tensor\shift_rows($in_place, $new, $in_place) === $in_place);
var_dump(values($in_place) === values($shifted));

/* out reuse */
$out = Toy\Tensor::zeros('f32', [4, 3]);
var_dump(Toy\Tensor\shift_rows($state, $new, $out) === $out);
var_dump($out->toBytes() === $shifted->toBytes());

/* One row: the new row is the whole state, and without it nothing changes */
$single = tensor([1, 4], [1.0, 2.0, 3.0, 4.0]);
var_dump(values(Toy\Tensor\shift_rows($single, tensor([4], [9.0, 8.0, 7.0, 6.0])))
    === [9.0, 8.0, 7.0, 6.0]);
var_dump(values(Toy\Tensor\shift_rows($single)) === [1.0, 2.0, 3.0, 4.0]);

/* A width that is not a multiple of eight, and the one value case */
$odd = tensor([3, 5], range(1, 15));
var_dump(values(Toy\Tensor\shift_rows($odd, tensor([5], [0.0, 0.0, 0.0, 0.0, 0.0])))
    === [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
var_dump(values(Toy\Tensor\shift_rows(tensor([2, 1], [1.0, 2.0]), tensor([1], [7.0])))
    === [2.0, 7.0]);

/* The state of one convolution layer */
$state_rows = 6144;
$state_taps = 4;
$big = Toy\Tensor::full('f32', [$state_taps, $state_rows], 1.0);
$big_row = Toy\Tensor::full('f32', [$state_rows], 2.0);
$big_shifted = Toy\Tensor\shift_rows($big, $big_row, $big);
var_dump($big_shifted === $big);
$big_values = values($big);
var_dump($big_values[0] === 1.0);
var_dump($big_values[($state_taps - 1) * $state_rows] === 2.0);
var_dump($big_values[count($big_values) - 1] === 2.0);

$shift_rows = new ReflectionFunction('Toy\\Tensor\\shift_rows');
var_dump($shift_rows->getNumberOfParameters() === 3);
var_dump($shift_rows->getNumberOfRequiredParameters() === 1);
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
done
