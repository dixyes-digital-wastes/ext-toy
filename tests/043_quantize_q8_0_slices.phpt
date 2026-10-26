--TEST--
quantize_q8_0 slices rows without copying and keeps the source intact
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

$isas = ['scalar'];
if (Toy\Tensor\kernel_stats()['avx2_available']) {
    $isas[] = 'avx2';
}

$ok = true;
$rows = 9;
$k = 96;
$values = [];
for ($i = 0; $i < $rows * $k; $i++) {
    $values[] = test_value($i, 7) * ($i % 11 + 1.0);
}
$x = test_tensor([$rows, $k], $values);
$sourceBytes = $x->toBytes();

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    $full = Toy\Tensor\quantize_q8_0($x);
    $blocksPerRow = intdiv($k, 32);

    // Every row range selects the same bytes as the matching rows of the full result
    for ($begin = 0; $begin < $rows; $begin++) {
        for ($end = $begin + 1; $end <= $rows; $end++) {
            $part = Toy\Tensor\quantize_q8_0($x, $begin, $end);
            $ok = $ok && $part->shape() === [$end - $begin, $k];
            $ok = $ok && $part->bytes() === ($end - $begin) * $blocksPerRow * 34;
            $ok = $ok && $part->toBytes()
                === substr($full->toBytes(), $begin * $blocksPerRow * 34,
                           ($end - $begin) * $blocksPerRow * 34);
        }
    }

    // A null rowEnd selects the tail, and the last row is reachable alone
    $ok = $ok && Toy\Tensor\quantize_q8_0($x, 7)->shape() === [2, $k];
    $ok = $ok && Toy\Tensor\quantize_q8_0($x, 8, 9)->shape() === [1, $k];
    $ok = $ok && Toy\Tensor\quantize_q8_0($x, 7, null)->toBytes()
        === substr($full->toBytes(), 7 * $blocksPerRow * 34, 2 * $blocksPerRow * 34);

    // The scalar fallback agrees on a slice as well
    Toy\Tensor\kernel_enable(false);
    $fallback = Toy\Tensor\quantize_q8_0($x, 2, 5);
    Toy\Tensor\kernel_enable(true);
    $ok = $ok && $fallback->toBytes()
        === substr($full->toBytes(), 2 * $blocksPerRow * 34, 3 * $blocksPerRow * 34);
}

// A 1-D tensor is one row, so the only valid range is the whole vector
$vector = test_tensor([$k], array_slice($values, 0, $k));
$ok = $ok && Toy\Tensor\quantize_q8_0($vector)->shape() === [$k];
$ok = $ok && Toy\Tensor\quantize_q8_0($vector, 0, 1)->shape() === [$k];
$ok = $ok && Toy\Tensor\quantize_q8_0($vector)->toBytes()
    === Toy\Tensor\quantize_q8_0(test_tensor([1, $k], array_slice($values, 0, $k)))->toBytes();

// The operator only reads the source
$ok = $ok && $x->toBytes() === $sourceBytes && $x->dtype === 'f32' && $x->shape() === [$rows, $k];

// Every call returns a new tensor with the same bytes
$first = Toy\Tensor\quantize_q8_0($x, 4, 5);
$second = Toy\Tensor\quantize_q8_0($x, 4, 5);
$ok = $ok && $first !== $second && $first->toBytes() === $second->toBytes();
$ok = $ok && $first->shape() === [1, $k];

var_dump($ok);
echo "done\n";
?>
--EXPECT--
bool(true)
done
