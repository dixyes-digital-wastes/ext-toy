--TEST--
Generated kernels match the scalar reference for both codegen ISAs
--EXTENSIONS--
toy
--FILE--
<?php
function tensor(array $shape, array $values): Toy\Tensor {
    return Toy\Tensor::fromBytes('f32', $shape, pack('g*', ...$values));
}
function ramp(int $count, int $start, int $step): array {
    $out = [];
    for ($i = 0; $i < $count; $i++) {
        $out[] = $start + $i * $step;
    }
    return $out;
}

$stats = Toy\Tensor\kernel_stats();
$isas = ['scalar'];
if ($stats['avx2_available']) {
    $isas[] = 'avx2';
}

$element_ok = true;
$matmul_ok = true;
$matvec_ok = true;

foreach ($isas as $isa) {
    Toy\Tensor\kernel_isa($isa);

    foreach ([1, 2, 3, 7, 8, 9, 15, 16, 17, 23, 64, 1024] as $count) {
        $a = tensor([$count], ramp($count, 1, 1));
        $b = tensor([$count], ramp($count, 2, 2));
        $w = tensor([$count], ramp($count, 3, 1));

        Toy\Tensor\kernel_enable(true);
        $add_kernel = Toy\Tensor\add($a, $b);
        $scale_kernel = Toy\Tensor\scale($a, 1.5);
        $norm_kernel = Toy\Tensor\rmsnorm($a, $w);
        $norm_zero_kernel = Toy\Tensor\rmsnorm($a, $w, 0.0);

        Toy\Tensor\kernel_enable(false);
        $add_scalar = Toy\Tensor\add($a, $b);
        $scale_scalar = Toy\Tensor\scale($a, 1.5);
        $norm_scalar = Toy\Tensor\rmsnorm($a, $w);
        $norm_zero_scalar = Toy\Tensor\rmsnorm($a, $w, 0.0);
        Toy\Tensor\kernel_enable(true);

        $element_ok = $element_ok
            && $add_kernel->toBytes() === $add_scalar->toBytes()
            && $scale_kernel->toBytes() === $scale_scalar->toBytes()
            && $norm_kernel->toBytes() === $norm_scalar->toBytes()
            && $norm_zero_kernel->toBytes() === $norm_zero_scalar->toBytes();
    }

    foreach ([[1, 1, 1], [2, 3, 4], [7, 5, 9], [8, 8, 8], [17, 15, 17]] as [$m, $n, $k]) {
        $a = tensor([$m, $k], ramp($m * $k, 1, 1));
        $b = tensor([$k, $n], ramp($k * $n, 1, 2));

        Toy\Tensor\kernel_enable(true);
        $kernel = Toy\Tensor\matmul_f32($a, $b, $m, $n, $k);
        Toy\Tensor\kernel_enable(false);
        $scalar = Toy\Tensor\matmul_f32($a, $b, $m, $n, $k);
        Toy\Tensor\kernel_enable(true);

        $matmul_ok = $matmul_ok && $kernel->toBytes() === $scalar->toBytes();
    }

    foreach ([[1, 1], [2, 3], [7, 5], [8, 8], [17, 15]] as [$m, $k]) {
        $w = tensor([$m, $k], ramp($m * $k, 1, 1));
        $x = tensor([$k], ramp($k, 1, 3));

        Toy\Tensor\kernel_enable(true);
        $kernel = Toy\Tensor\matvec_f32($w, $x, $m, $k);
        Toy\Tensor\kernel_enable(false);
        $scalar = Toy\Tensor\matvec_f32($w, $x, $m, $k);
        Toy\Tensor\kernel_enable(true);

        $matvec_ok = $matvec_ok && $kernel->toBytes() === $scalar->toBytes();
    }
}

Toy\Tensor\kernel_isa('avx2');
var_dump($element_ok);
var_dump($matmul_ok);
var_dump($matvec_ok);

$stats = Toy\Tensor\kernel_stats();
var_dump($stats['generated'] > 0);
var_dump($stats['scalar_calls'] > 0);
echo "done\n";
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
done
