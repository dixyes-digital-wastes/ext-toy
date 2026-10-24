--TEST--
f32 matvec rows are bit-identical for every m and row offset
--EXTENSIONS--
toy
--FILE--
<?php
function values(int $seed, int $count): array {
    $state = $seed;
    $out = [];
    for ($i = 0; $i < $count; $i++) {
        $state = ($state * 1103515245 + 12345) & 0x7fffffff;
        $out[] = $state / 0x7fffffff * 2.0 - 1.0;
    }
    return $out;
}

function tensorRows(array $rows, int $k): Toy\Tensor {
    $bytes = '';
    foreach ($rows as $row) {
        $bytes .= pack('g*', ...$row);
    }
    return Toy\Tensor::fromBytes('f32', [count($rows), $k], $bytes);
}

function window(array $rows, int $start, int $m, int $k): Toy\Tensor {
    return tensorRows(array_slice($rows, $start, $m), $k);
}

const ROWS = 24;
const STARTS = [0, 1, 7];
const WIDTHS = [1, 2, 7, 8, 9, 16, 17];

$stats = Toy\Tensor\kernel_stats();
$isas = ['scalar'];
if ($stats['avx2_available']) {
    $isas[] = 'avx2';
}

$cases = 0;
$divergent = 0;
$allRows = 0;

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    foreach ([5, 1021, 1024] as $k) {
        $rows = [];
        for ($r = 0; $r < ROWS; $r++) {
            $rows[] = values(20261024 + $r, $k);
        }
        $x = Toy\Tensor::fromBytes('f32', [$k], pack('g*', ...values(777, $k)));

        $seen = [];
        foreach (WIDTHS as $m) {
            foreach (STARTS as $start) {
                if ($start + $m > ROWS) {
                    continue;
                }
                $y = Toy\Tensor\matvec_f32(window($rows, $start, $m, $k), $x, $m, $k);
                $bytes = $y->toBytes();
                $cases++;
                for ($g = $start; $g < $start + $m; $g++) {
                    $seen[$g][bin2hex(substr($bytes, ($g - $start) * 4, 4))] = true;
                }
            }
        }
        foreach ($seen as $row => $hexes) {
            if (count($hexes) > 1) {
                $divergent++;
                echo "$isa k=$k row $row: " . implode(' ', array_keys($hexes)) . "\n";
            }
        }
        $allRows += count($seen);
    }
}

// Above the emitter cap the scalar fallback takes over, while a small window
// still uses the generated kernel. The shared rows must stay equal
$k = 8;
$blockRows = 64;
$block = [];
for ($r = 0; $r < $blockRows; $r++) {
    $block[] = values(9000 + $r, $k);
}
$blockBytes = tensorRows($block, $k)->toBytes();
$bigRows = 65536 + 64;
$bigBytes = substr(str_repeat($blockBytes, intdiv($bigRows, $blockRows) + 1), 0, $bigRows * $k * 4);
$big = Toy\Tensor::fromBytes('f32', [$bigRows, $k], $bigBytes);
$small = Toy\Tensor::fromBytes('f32', [$blockRows, $k], $blockBytes);
$x8 = Toy\Tensor::fromBytes('f32', [$k], pack('g*', ...values(4242, $k)));
$bigY = Toy\Tensor\matvec_f32($big, $x8, $bigRows, $k);
$smallY = Toy\Tensor\matvec_f32($small, $x8, $blockRows, $k);

var_dump($divergent === 0);
var_dump($cases > 0);
var_dump($allRows > 0);
var_dump(substr($bigY->toBytes(), 0, $blockRows * 4) === $smallY->toBytes());
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
done
