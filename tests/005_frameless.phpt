--TEST--
Frameless handlers produce the same bytes as the zif path
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function values(int $count, int $seed): array {
    $out = [];
    for ($i = 0; $i < $count; $i++) {
        $out[] = (($i * 7 + $seed) % 13) * 0.25;
    }
    return $out;
}
function call_dynamic(string $name, array $args): Toy\Tensor {
    $fn = 'Toy\\Tensor\\' . $name;
    return $fn(...$args);
}

$ok = true;
foreach ([1, 2, 3, 7, 8, 9, 15, 16, 17, 64, 1024] as $count) {
    $a = tensor([$count], values($count, 1));
    $b = tensor([$count], values($count, 2));
    $w = tensor([$count], values($count, 3));

    $ok = $ok && Toy\Tensor\add($a, $b)->toBytes() === call_dynamic('add', [$a, $b])->toBytes();
    $ok = $ok && Toy\Tensor\scale($a, 1.5)->toBytes() === call_dynamic('scale', [$a, 1.5])->toBytes();
    $ok = $ok && Toy\Tensor\rmsnorm($a, $w)->toBytes() === call_dynamic('rmsnorm', [$a, $w])->toBytes();
    $ok = $ok && Toy\Tensor\rmsnorm($a, $w, 1e-6)->toBytes()
        === call_dynamic('rmsnorm', [$a, $w, 1e-6])->toBytes();
}
var_dump($ok);

$stats = Toy\Tensor\kernel_stats();
var_dump($stats['flf_calls'] > 0);
var_dump($stats['zif_calls'] > 0);
var_dump($stats['generated'] > 0);
var_dump($stats['kernel_calls'] >= $stats['generated']);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
