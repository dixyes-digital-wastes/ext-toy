--TEST--
quantize_q8_0 matches the byte reference, the cast and both codegen paths
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
$lengths = [32, 64, 96, 128, 1024, 3584];

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);

    // Random values of both signs, one tensor per length
    foreach ($lengths as $k) {
        $values = [];
        for ($i = 0; $i < $k; $i++) {
            $values[] = test_value($i, $k % 17 + 1) * ($i % 3 + 1.0);
        }
        $x = test_tensor([$k], $values);
        $got = Toy\Tensor\quantize_q8_0($x);
        $ok = $ok && $got->dtype === 'q8_0' && $got->shape() === [$k];
        $ok = $ok && $got->bytes() === intdiv($k, 32) * 34;
        $ok = $ok && $got->toBytes() === test_q8_quantize($values);
        $ok = $ok && $got->toBytes() === $x->cast('q8_0')->toBytes();
    }

    // Rows, so the last dimension is the row width
    foreach ([[1, 32], [3, 64], [7, 96], [5, 1024]] as [$rows, $k]) {
        $values = [];
        for ($i = 0; $i < $rows * $k; $i++) {
            $values[] = test_value($i, $rows + $k) * 5.0;
        }
        $x = test_tensor([$rows, $k], $values);
        $got = Toy\Tensor\quantize_q8_0($x);
        $ok = $ok && $got->shape() === [$rows, $k];
        $ok = $ok && $got->toBytes() === test_q8_quantize($values);
        $ok = $ok && $got->toBytes() === $x->cast('q8_0')->toBytes();
    }
}

/*
 * Fixed blocks: zero, a zero scale from NaNs, exact halves that round away
 * from zero, infinite values, subnormal and tiny magnitudes, and a scale that
 * overflows f16
 */
$fixed = [];
$blocks = [
    array_fill(0, 32, 0.0),
    array_fill(0, 32, NAN),
    array_map(fn (int $i): float => $i % 2 === 0 ? INF : -INF, range(0, 31)),
    array_map(fn (int $i): float => $i === 5 ? INF : (float)($i - 16), range(0, 31)),
    array_map(fn (int $i): float => $i === 0 ? 1.0 : NAN, range(0, 31)),
    array_fill(0, 32, 1.401298464324817E-45),
    array_fill(0, 32, 1.1754943508222875E-38),
    array_fill(0, 32, 3.4028234663852886E+38),
    array_map(fn (int $i): float => (float)(2 * ($i - 16) + 1), range(0, 31)),
    array_map(fn (int $i): float => (float)($i - 16) + 0.5, range(0, 31)),
];
$blocks[8][0] = 254.0;
$blocks[9][16] = 127.0;
foreach ($blocks as $block) {
    foreach ($block as $value) {
        $fixed[] = $value;
    }
}

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);
    $tensor = test_tensor([count($blocks), 32], $fixed);
    $got = Toy\Tensor\quantize_q8_0($tensor);
    $ok = $ok && $got->toBytes() === test_q8_quantize($fixed);

    // The kernel and the scalar fallback agree bit for bit
    Toy\Tensor\kernel_enable(false);
    $fallback = Toy\Tensor\quantize_q8_0($tensor);
    Toy\Tensor\kernel_enable(true);
    $ok = $ok && $fallback->toBytes() === $got->toBytes();
}

// A power of two scale makes every product exact, so the double precision
// variant of the demo PHP code produces the same bytes there
$exact = [];
foreach ([0, 1, 2, 3, 5, 8, 9] as $index) {
    foreach ($blocks[$index] as $value) {
        $exact[] = $value;
    }
}
$exactTensor = test_tensor([intdiv(count($exact), 32), 32], $exact);
$ok = $ok && Toy\Tensor\quantize_q8_0($exactTensor)->toBytes() === test_q8_quantize_double($exact);

// The scale bytes of the non-finite blocks: Inf stays Inf, a zero block stays zero
$bytes = Toy\Tensor\quantize_q8_0(test_tensor([count($blocks), 32], $fixed))->toBytes();
$ok = $ok && unpack('v', substr($bytes, 34, 2))[1] === 0x0000;
$ok = $ok && unpack('v', substr($bytes, 2 * 34, 2))[1] === 0x7C00;
$ok = $ok && unpack('v', substr($bytes, 3 * 34, 2))[1] === 0x7C00;
$ok = $ok && substr($bytes, 34 * 6 + 2, 32) === str_repeat("\x00", 32);

// A 3584 wide row covers the FFN down projection shape
$ffn = test_tensor([3584], array_map(fn (int $i): float => test_value($i, 11), range(0, 3583)));
$ok = $ok && Toy\Tensor\quantize_q8_0($ffn)->bytes() === 112 * 34;

// The function is a plain zif handler with two optional arguments
$fn = new ReflectionFunction('Toy\\Tensor\\quantize_q8_0');
$ok = $ok && $fn->getNumberOfParameters() === 3 && $fn->getNumberOfRequiredParameters() === 1;
$ok = $ok && $fn->getParameters()[1]->isOptional()
    && $fn->getParameters()[1]->getDefaultValue() === 0;
$ok = $ok && $fn->getParameters()[2]->isOptional()
    && $fn->getParameters()[2]->getDefaultValue() === null;
$ok = $ok && $fn->getReturnType()?->getName() === 'Toy\\Tensor';

// One call counts one zif, and an AVX2 host resolves a generated kernel
Toy\Tensor\kernel_isa($isas[count($isas) - 1]);
Toy\Tensor\kernel_stats(true);
$input = test_tensor([1024], array_map(fn (int $i): float => test_value($i, 3), range(0, 1023)));
Toy\Tensor\quantize_q8_0($input);
$stats = Toy\Tensor\kernel_stats();
$ok = $ok && $stats['zif_calls'] === 1 && $stats['flf_calls'] === 0;
if (count($isas) > 1) {
    $ok = $ok && $stats['kernel_calls'] === 1 && $stats['scalar_calls'] === 0;
} else {
    $ok = $ok && $stats['kernel_calls'] === 0 && $stats['scalar_calls'] === 1;
}

var_dump($ok);
var_dump(count($isas) >= 1);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
done
