--TEST--
q8_0 x q8_0 matvec matches the exact reference and the scalar fallback
--EXTENSIONS--
toy
--FILE--
<?php
require __DIR__ . '/helpers/q8.php';

$isas = ['scalar'];
if (Toy\Tensor\kernel_stats()['avx2_available']) {
    $isas[] = 'avx2';
}

/*
 * d = 2^-4 for the weights and 2^-3 for the activation, and the bytes stay in
 * [-7, 7], so every product, fma and sum is exact in f32
 */
$dw = 0x2C00;
$dx = 0x3000;
$ok = true;
$exact = true;

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);

    foreach ([[5, 64], [17, 256]] as [$m, $k]) {
        $blocks = intdiv($k, 32);
        $wblocks = [];
        for ($i = 0; $i < $m * $blocks; $i++) {
            $wblocks[] = ['d' => $dw, 'q' => test_q8_small(32, 7, $i + 1)];
        }
        $xblocks = [];
        for ($b = 0; $b < $blocks; $b++) {
            $xblocks[] = ['d' => $dx, 'q' => test_q8_small(32, 7, $b + 3)];
        }
        $w = Toy\Tensor::fromBytes('q8_0', [$m, $k], test_q8_pack($wblocks));
        $x = Toy\Tensor::fromBytes('q8_0', [$k], test_q8_pack($xblocks));

        $got = array_values(unpack('g*', Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k)->toBytes()));
        for ($i = 0; $i < $m; $i++) {
            $expected = test_q8_q8_dot($wblocks, $i, $xblocks, $blocks);
            $exact = $exact && $got[$i] === $expected;
        }

        // The scalar path must agree with the kernel bit for bit
        Toy\Tensor\kernel_enable(false);
        $fallback = Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k);
        Toy\Tensor\kernel_enable(true);
        $ok = $ok && $fallback->toBytes() === Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k)->toBytes();
    }
}

// Arbitrary bytes and scales: the two paths still agree bit for bit
$m = 9;
$k = 128;
$blocks = intdiv($k, 32);
$wblocks = [];
for ($i = 0; $i < $m * $blocks; $i++) {
    $wblocks[] = ['d' => 0x3000 + $i * 7 % 0x200, 'q' => test_q8_small(32, 120, $i + 5)];
}
$xblocks = [];
for ($b = 0; $b < $blocks; $b++) {
    $xblocks[] = ['d' => 0x2C00 + $b * 11, 'q' => test_q8_small(32, 127, $b + 9)];
}
$w = Toy\Tensor::fromBytes('q8_0', [$m, $k], test_q8_pack($wblocks));
$x = Toy\Tensor::fromBytes('q8_0', [$k], test_q8_pack($xblocks));
$kernel = Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k);
Toy\Tensor\kernel_enable(false);
$scalar = Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k);
Toy\Tensor\kernel_enable(true);
$ok = $ok && $kernel->toBytes() === $scalar->toBytes();

// Two -128 bytes follow the ggml sign trick: -16384, not +16384
$edge = Toy\Tensor::fromBytes('q8_0', [1, 32],
    test_q8_pack([['d' => 0x3C00, 'q' => array_fill(0, 32, -128)]]));
$edgeX = Toy\Tensor::fromBytes('q8_0', [32],
    test_q8_pack([['d' => 0x3C00, 'q' => array_fill(0, 32, -128)]]));
$edgeOut = array_values(unpack('g*', Toy\Tensor\matvec_q8_0_q8_0($edge, $edgeX, 1, 32)->toBytes()));
$ok = $ok && $edgeOut[0] === -16384.0 * 32;

// out reuse returns the same object with the same bytes
$out = Toy\Tensor::zeros('f32', [$m]);
$reused = Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k, $out);
$ok = $ok && $reused === $out && $out->toBytes() === $kernel->toBytes();

// The ops are plain zif handlers
Toy\Tensor\kernel_stats(true);
$fn = new ReflectionFunction('Toy\\Tensor\\matvec_q8_0_q8_0');
$ok = $ok && $fn->getNumberOfParameters() === 5 && $fn->getNumberOfRequiredParameters() === 4;
$ok = $ok && $fn->getParameters()[4]->isOptional() && $fn->getParameters()[4]->getDefaultValue() === null;
Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k);
$stats = Toy\Tensor\kernel_stats();
$ok = $ok && $stats['zif_calls'] === 1 && $stats['flf_calls'] === 0;

// The scalar ISA has no generated kernel for this op, so it counts a scalar call
if (count($isas) > 1) {
    Toy\Tensor\kernel_isa('scalar');
    Toy\Tensor\kernel_stats(true);
    $viaScalar = Toy\Tensor\matvec_q8_0_q8_0($w, $x, $m, $k);
    $stats = Toy\Tensor\kernel_stats();
    $ok = $ok && $stats['kernel_calls'] === 0 && $stats['scalar_calls'] === 1;
    $ok = $ok && $viaScalar->toBytes() === $kernel->toBytes();
}

var_dump($exact);
var_dump($ok);
var_dump(count($isas) >= 1);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
done
