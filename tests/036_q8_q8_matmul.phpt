--TEST--
q8_0 x q8_0 matmul handles many columns and matches one matvec per column
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

$isas = ['scalar'];
if (Toy\Tensor\kernel_stats()['avx2_available']) {
    $isas[] = 'avx2';
}

// Small bytes and power-of-two scales keep the whole chain exact
$dw = 0x2C00;
$dx = 0x3000;
$m = 9;
$n = 32;
$k = 64;
$blocks = intdiv($k, 32);

$wblocks = [];
for ($i = 0; $i < $m * $blocks; $i++) {
    $wblocks[] = ['d' => $dw, 'q' => test_q8_small(32, 7, $i + 2)];
}
// One q8_0 vector of k values per column, laid out one column after another
$columns = [];
for ($j = 0; $j < $n; $j++) {
    $column = [];
    for ($b = 0; $b < $blocks; $b++) {
        $column[] = ['d' => $dx, 'q' => test_q8_small(32, 7, $j * $blocks + $b + 4)];
    }
    $columns[$j] = $column;
}
$w = Toy\Tensor::fromBytes('q8_0', [$m, $k], test_q8_pack($wblocks));
$x = Toy\Tensor::fromBytes('q8_0', [$k, $n], test_q8_pack(array_merge(...$columns)));

$exact = true;
$ok = true;
$perColumn = true;

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);

    $values = array_values(unpack('g*', Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k)->toBytes()));
    for ($i = 0; $i < $m; $i++) {
        for ($j = 0; $j < $n; $j++) {
            $expected = test_q8_q8_dot($wblocks, $i, $columns[$j], $blocks);
            $exact = $exact && $values[$i * $n + $j] === $expected;
        }
    }

    // Column j of the matmul is the matvec of the same bytes
    for ($j = 0; $j < $n; $j += 7) {
        $vector = Toy\Tensor::fromBytes('q8_0', [$k], test_q8_pack($columns[$j]));
        $single = array_values(unpack('g*', Toy\Tensor\matvec_q8_0_q8_0($w, $vector, $m, $k)->toBytes()));
        for ($i = 0; $i < $m; $i++) {
            $perColumn = $perColumn && $single[$i] === $values[$i * $n + $j];
        }
    }

    Toy\Tensor\kernel_enable(false);
    $fallback = Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k);
    Toy\Tensor\kernel_enable(true);
    $ok = $ok && $fallback->toBytes() === Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $n, $k)->toBytes();
}

/*
 * Arbitrary bytes and scales over several widths, so the column loop and the
 * row group tail of the kernel both run
 */
$m = 17;
$k = 128;
$blocks = intdiv($k, 32);
$wblocks = [];
for ($i = 0; $i < $m * $blocks; $i++) {
    $wblocks[] = ['d' => 0x3000 + ($i * 13) % 0x400, 'q' => test_q8_small(32, 127, $i + 1)];
}
$w = Toy\Tensor::fromBytes('q8_0', [$m, $k], test_q8_pack($wblocks));

foreach ([32, 64, 96] as $wide) {
    $columns = [];
    for ($j = 0; $j < $wide; $j++) {
        $column = [];
        for ($b = 0; $b < $blocks; $b++) {
            $column[] = ['d' => 0x2C00 + ($j * 5 + $b * 3) % 0x300,
                'q' => test_q8_small(32, 127, $j * 3 + $b + 7)];
        }
        $columns[] = $column;
    }
    $x = Toy\Tensor::fromBytes('q8_0', [$k, $wide], test_q8_pack(array_merge(...$columns)));
    Toy\Tensor\kernel_isa('avx2');
    $kernel = Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $wide, $k);
    Toy\Tensor\kernel_enable(false);
    $scalar = Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $wide, $k);
    Toy\Tensor\kernel_enable(true);
    $ok = $ok && $kernel->toBytes() === $scalar->toBytes();

    $out = Toy\Tensor::zeros('f32', [$m, $wide]);
    $reused = Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, $wide, $k, $out);
    $ok = $ok && $reused === $out && $out->toBytes() === $kernel->toBytes();
}

$fn = new ReflectionFunction('Toy\\Tensor\\matmul_q8_0_q8_0');
$ok = $ok && $fn->getNumberOfParameters() === 6 && $fn->getNumberOfRequiredParameters() === 5;
$ok = $ok && $fn->getParameters()[5]->isOptional() && $fn->getParameters()[5]->getDefaultValue() === null;

Toy\Tensor\kernel_stats(true);
Toy\Tensor\matmul_q8_0_q8_0($w, $x, $m, 96, $k);
$stats = Toy\Tensor\kernel_stats();
$ok = $ok && $stats['zif_calls'] === 1 && $stats['flf_calls'] === 0;

var_dump($exact);
var_dump($perColumn);
var_dump($ok);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
done
