--TEST--
Tensor memo hits stay valid across reshape and a cast invalidates them
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function values(int $count, float $step): array {
    $out = [];
    for ($i = 0; $i < $count; $i++) {
        $out[] = $i * $step;
    }
    return $out;
}
function delta(array $before, array $after, string $field): int {
    return $after[$field] - $before[$field];
}
function show(string $label, array $before, array $after): void {
    echo $label,
        ' hits=', delta($before, $after, 'memo_hits'),
        ' misses=', delta($before, $after, 'memo_misses'),
        ' generated=', delta($before, $after, 'generated'),
        ' arena_hits=', delta($before, $after, 'arena_hits'), "\n";
}

$a = tensor([4, 4], values(16, 0.5));
$b = tensor([4, 4], values(16, 0.25));

Toy\Tensor\kernel_stats(true);
Toy\Tensor\add($a, $b);
$s1 = Toy\Tensor\kernel_stats();
show('first', array_fill_keys(
    ['memo_hits', 'memo_misses', 'generated', 'arena_hits'], 0), $s1);

Toy\Tensor\add($a, $b);
$s2 = Toy\Tensor\kernel_stats();
show('second', $s1, $s2);

$a->reshape([2, 8]);
$b->reshape([2, 8]);
Toy\Tensor\add($a, $b);
$s3 = Toy\Tensor\kernel_stats();
show('reshape', $s2, $s3);

$a2 = $a->cast('f32');
Toy\Tensor\add($a2, $b);
$s4 = Toy\Tensor\kernel_stats();
show('cast', $s3, $s4);

Toy\Tensor\scale($a, 2.0);
$s5 = Toy\Tensor\kernel_stats();
show('scale first', $s4, $s5);

Toy\Tensor\scale($a, 2.0);
$s6 = Toy\Tensor\kernel_stats();
show('scale again', $s5, $s6);

Toy\Tensor\scale($a, 4.0);
$s7 = Toy\Tensor\kernel_stats();
show('scale other factor', $s6, $s7);

$x = tensor([8], values(8, 0.5));
$w = tensor([8], values(8, 0.125));
Toy\Tensor\kernel_stats(true);
Toy\Tensor\rmsnorm($x, $w);
$n1 = Toy\Tensor\kernel_stats();
show('rmsnorm first', array_fill_keys(
    ['memo_hits', 'memo_misses', 'generated', 'arena_hits'], 0), $n1);

Toy\Tensor\rmsnorm($x, $w);
$n2 = Toy\Tensor\kernel_stats();
show('rmsnorm again', $n1, $n2);

Toy\Tensor\rmsnorm($x, $w, 0.0);
$n3 = Toy\Tensor\kernel_stats();
show('rmsnorm other eps', $n2, $n3);

Toy\Tensor\rmsnorm($x, $w, 0.0);
$n4 = Toy\Tensor\kernel_stats();
show('rmsnorm other eps again', $n3, $n4);

$before = Toy\Tensor\add($a, $b);
Toy\Tensor\kernel_enable(false);
$scalar = Toy\Tensor\add($a, $b);
Toy\Tensor\kernel_enable(true);
var_dump($before->toBytes() === $scalar->toBytes());
echo "done\n";
?>
--EXPECT--
first hits=0 misses=1 generated=1 arena_hits=0
second hits=1 misses=0 generated=0 arena_hits=0
reshape hits=1 misses=0 generated=0 arena_hits=0
cast hits=0 misses=1 generated=0 arena_hits=1
scale first hits=0 misses=1 generated=1 arena_hits=0
scale again hits=1 misses=0 generated=0 arena_hits=0
scale other factor hits=0 misses=1 generated=1 arena_hits=0
rmsnorm first hits=0 misses=1 generated=1 arena_hits=0
rmsnorm again hits=1 misses=0 generated=0 arena_hits=0
rmsnorm other eps hits=0 misses=1 generated=1 arena_hits=0
rmsnorm other eps again hits=1 misses=0 generated=0 arena_hits=0
bool(true)
done
