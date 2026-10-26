--TEST--
gdn_recurrence rejects wrong dtypes, shapes, slices and aliased outputs
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}
function gdn(Toy\Tensor $q, Toy\Tensor $k, Toy\Tensor $v, Toy\Tensor $decay,
             Toy\Tensor $beta, Toy\Tensor $state, int $headBegin, int $headEnd,
             int $colBegin, int $colEnd, float $scale = 1.0,
             ?Toy\Tensor $out = null): Toy\Tensor {
    return Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state,
        $headBegin, $headEnd, $colBegin, $colEnd, $scale, $out);
}

$q = tensor([2, 3], [0.5, -0.5, 0.25, 0.75, -0.25, 0.125]);
$k = tensor([2, 3], [0.25, 0.5, -0.75, 0.125, -0.5, 0.25]);
$v = tensor([2, 4], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
$decay = tensor([2], [0.9, 0.8]);
$beta = tensor([2], [0.5, 0.4]);
$state = tensor([2, 4, 3], array_map(fn(int $i): float => ($i % 7) / 8, range(0, 23)));
$out = Toy\Tensor::zeros('f32', [2, 4]);
$q8 = tensor([2, 32], array_fill(0, 64, 1.0))->cast('q8_0');
$stateBytes = $state->toBytes();

// dtypes
show(fn() => gdn($q8, $k, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $q8, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $q8, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $q8, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $q8, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, $q8, 0, 2, 0, 4));

// ranks and shapes
$flat = tensor([6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
$cube = tensor([2, 3, 1], array_fill(0, 6, 1.0));
$wide = tensor([3, 2], array_fill(0, 6, 1.0));
$short = tensor([8], array_fill(0, 8, 1.0));
show(fn() => gdn($flat, $k, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($cube, $k, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $wide, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $short, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, tensor([3, 4], array_fill(0, 12, 1.0)),
    $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, tensor([2, 1], [0.9, 0.8]), $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, tensor([1], [0.9]), $beta, $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, tensor([1], [0.5]), $state, 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, tensor([2, 12], array_fill(0, 24, 1.0)), 0, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, tensor([2, 3, 4], array_fill(0, 24, 1.0)), 0, 2, 0, 4));

// slice bounds
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, -1, 2, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 3, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, -1, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 5));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 2, 1, 0, 4));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 3, 2));

// out tensor: dtype, shape and aliasing
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 4, 1.0, $q8));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 4, 1.0,
    tensor([2, 5], array_fill(0, 10, 0.0))));
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 4, 1.0, $v));

$q4 = tensor([2, 4], array_fill(0, 8, 0.5));
$k4 = tensor([2, 4], array_fill(0, 8, 0.25));
$state4 = tensor([2, 4, 4], array_fill(0, 32, 0.0));
$out4 = Toy\Tensor::zeros('f32', [2, 4]);
show(fn() => gdn($q4, $k4, $v, $decay, $beta, $state4, 0, 2, 0, 4, 1.0, $q4));
// The same call without aliasing succeeds and returns the out tensor
var_dump(gdn($q4, $k4, $v, $decay, $beta, $state4, 0, 2, 0, 4, 1.0, $out4) === $out4);

// argument types and counts
show(fn() => Toy\Tensor\gdn_recurrence(1, $k, $v, $decay, $beta, $state, 0, 2, 0, 4));
show(fn() => Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state, 'x', 2, 0, 4));
show(fn() => Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 4, []));
show(fn() => Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state, 0, 2, 0));
show(fn() => Toy\Tensor\gdn_recurrence($q, $k, $v, $decay, $beta, $state,
    0, 2, 0, 4, 1.0, null, 1));

var_dump($state->toBytes() === $stateBytes);
var_dump($out->toBytes() === str_repeat("\0", 32));
// A rejected call must not touch the state or the out tensor
show(fn() => gdn($q, $k, $v, $decay, $beta, $state, 0, 2, 0, 4, 1.0, $v));
var_dump($state->toBytes() === $stateBytes);
var_dump($v->toBytes() === tensor([2, 4], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])->toBytes());
echo "done\n";
?>
--EXPECT--
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
ValueError
bool(true)
TypeError
TypeError
TypeError
ArgumentCountError
ArgumentCountError
bool(true)
bool(true)
ValueError
bool(true)
bool(true)
done
