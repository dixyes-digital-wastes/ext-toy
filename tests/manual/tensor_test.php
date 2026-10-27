<?php

declare(strict_types=1);

/*
 * PHP-level tests for the tensor core and the five functions; run with the
 * built extension:
 *   php -n -d extension=$PWD/modules/toy.so tests/manual/tensor_test.php
 * The C-only alias API is covered by tests/c/tensor_test.c
 * phpize --clean deletes tests/*.php, so this script lives in a subdirectory
 */

$failures = 0;
$checks = 0;

function check(string $label, bool $condition, string $detail = ''): void
{
    global $failures, $checks;
    $checks++;
    if ($condition) {
        echo "ok - {$label}\n";
        return;
    }
    $failures++;
    $suffix = $detail === '' ? '' : ": {$detail}";
    fwrite(STDERR, "not ok - {$label}{$suffix}\n");
}

function check_close(string $label, float $actual, float $expected, float $tolerance = 1e-6): void
{
    check($label, abs($actual - $expected) <= $tolerance,
        "got {$actual}, expected {$expected}");
}

function check_throws(string $label, string $class, callable $callback): void
{
    try {
        $callback();
    } catch (Throwable $error) {
        check($label, $error instanceof $class,
            get_class($error) . ': ' . $error->getMessage());
        return;
    }
    check($label, false, 'no exception');
}

function floats_to_bytes(array $values): string
{
    return pack('g*', ...$values);
}

function bytes_to_floats(string $bytes): array
{
    return array_values(unpack('g*', $bytes));
}

function tensor_from_floats(string $dtype, array $shape, array $values): Toy\Tensor
{
    return Toy\Tensor::fromBytes($dtype, $shape, floats_to_bytes($values));
}

// Class shape
check('class is registered', class_exists(Toy\Tensor::class));
$class = new ReflectionClass(Toy\Tensor::class);
check('class is final', $class->isFinal());
$dtype_property = $class->getProperty('dtype');
check('dtype is public', $dtype_property->isPublic());
check('dtype is readonly', $dtype_property->isReadOnly());
check('dtype is typed', (string) $dtype_property->getType() === 'string');
check('alias is not exposed to PHP',
    !function_exists('Toy\\Tensor\\alias') && !method_exists(Toy\Tensor::class, 'alias'));

// Construction and metadata
$zeros = Toy\Tensor::zeros('f32', [2, 3]);
check('zeros dtype', $zeros->dtype === 'f32');
check('zeros shape', $zeros->shape() === [2, 3]);
check('zeros ndim', $zeros->ndim() === 2);
check('zeros bytes', $zeros->bytes() === 24);
check('zeros data is zero', $zeros->toBytes() === str_repeat("\0", 24));

$ones = Toy\Tensor::ones('f32', [4]);
check('ones shape', $ones->shape() === [4]);
check('ones data is one', bytes_to_floats($ones->toBytes()) === [1.0, 1.0, 1.0, 1.0]);

$full = Toy\Tensor::full('f32', [2, 2], 0.5);
check('full shape', $full->shape() === [2, 2]);
check('full data', bytes_to_floats($full->toBytes()) === [0.5, 0.5, 0.5, 0.5]);

$from_bytes = Toy\Tensor::fromBytes('f32', [2], floats_to_bytes([1.0, -2.5]));
check('fromBytes shape', $from_bytes->shape() === [2]);
check('fromBytes keeps bytes', $from_bytes->toBytes() === floats_to_bytes([1.0, -2.5]));

$int_full = Toy\Tensor::full('f32', [1], 7);
check('full accepts integer', bytes_to_floats($int_full->toBytes()) === [7.0]);

// reshape
$reshape = Toy\Tensor::fromBytes('f32', [2, 3], floats_to_bytes(range(1, 6)));
$bytes_before = $reshape->toBytes();
$same = $reshape->reshape([3, 2]);
check('reshape returns the same object', $same === $reshape);
check('reshape updates shape', $reshape->shape() === [3, 2]);
check('reshape keeps bytes', $reshape->toBytes() === $bytes_before);
check('reshape updates ndim', $reshape->ndim() === 2);
check('reshape to 1-D', $reshape->reshape([6])->shape() === [6]);

// cast
$cast_source = tensor_from_floats('f32', [2, 2], [1.0, 2.0, 3.0, 4.0]);
$cast = $cast_source->cast('f32');
check('cast returns a new object', $cast !== $cast_source);
check('cast keeps dtype', $cast->dtype === 'f32');
check('cast keeps shape', $cast->shape() === [2, 2]);
check('cast copies data', $cast->toBytes() === $cast_source->toBytes());
$cast_source->reshape([4]);
check('cast is independent of shape changes', $cast->shape() === [2, 2]);

// Errors
check_throws('rejects an unknown dtype', ValueError::class,
    fn () => Toy\Tensor::zeros('f64', [1]));
$q8_zeros = Toy\Tensor::zeros('q8_0', [32]);
check('accepts whole q8_0 blocks',
    $q8_zeros->dtype === 'q8_0' && $q8_zeros->bytes() === 34);
check_throws('rejects an empty shape', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', []));
check_throws('rejects five dimensions', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', [1, 2, 3, 4, 5]));
check_throws('rejects a zero dimension', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', [0]));
check_throws('rejects a negative dimension', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', [-1]));
check_throws('rejects a non-integer dimension', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', [1.5]));
check_throws('rejects a non-list shape', ValueError::class,
    fn () => Toy\Tensor::zeros('f32', [1 => 1, 0 => 2]));
check_throws('rejects a short byte string', ValueError::class,
    fn () => Toy\Tensor::fromBytes('f32', [2, 2], str_repeat("\0", 15)));
check_throws('rejects a long byte string', ValueError::class,
    fn () => Toy\Tensor::fromBytes('f32', [2, 2], str_repeat("\0", 17)));
check_throws('reshape rejects a different byte count', ValueError::class,
    fn () => $reshape->reshape([7]));
check_throws('cast rejects an unsupported conversion', ValueError::class,
    fn () => $cast_source->cast('q8_0'));
check_throws('dtype is immutable', Error::class,
    function () use ($zeros): void {
        $zeros->dtype = 'q8_0';
    });
check_throws('constructor is private', Error::class, fn () => new Toy\Tensor());
check_throws('cloning is not supported', Error::class, fn () => clone $zeros);
check_throws('dynamic properties are rejected', Error::class,
    function () use ($zeros): void {
        $zeros->extra = 1;
    });
check_throws('serialization is not supported', Exception::class, fn () => serialize($zeros));

// The engine refuses to bypass the constructor of a final internal class
check_throws('constructor cannot be bypassed', ReflectionException::class,
    fn () => (new ReflectionClass(Toy\Tensor::class))->newInstanceWithoutConstructor());

// Reference operations
$a = tensor_from_floats('f32', [2, 2], [1.0, 2.0, 3.0, 4.0]);
$b = tensor_from_floats('f32', [2, 2], [5.0, 6.0, 7.0, 8.0]);

$matmul = Toy\Tensor\matmul_f32($a, $b, 2, 2, 2);
check('matmul shape', $matmul->shape() === [2, 2]);
check('matmul dtype', $matmul->dtype === 'f32');
$matmul_values = bytes_to_floats($matmul->toBytes());
foreach ([19.0, 22.0, 43.0, 50.0] as $index => $expected) {
    check_close("matmul value {$index}", $matmul_values[$index], $expected);
}

$vector = tensor_from_floats('f32', [2], [5.0, 6.0]);
$matvec = Toy\Tensor\matvec_f32($a, $vector, 2, 2);
check('matvec shape', $matvec->shape() === [2]);
$matvec_values = bytes_to_floats($matvec->toBytes());
check_close('matvec value 0', $matvec_values[0], 17.0);
check_close('matvec value 1', $matvec_values[1], 39.0);

$added = Toy\Tensor\add($a, $b);
check('add shape', $added->shape() === [2, 2]);
check('add values', bytes_to_floats($added->toBytes()) === [6.0, 8.0, 10.0, 12.0]);

$scaled = Toy\Tensor\scale($a, 2.5);
check('scale shape', $scaled->shape() === [2, 2]);
check('scale values', bytes_to_floats($scaled->toBytes()) === [2.5, 5.0, 7.5, 10.0]);

$norm_x = tensor_from_floats('f32', [2], [3.0, 4.0]);
$norm_w = tensor_from_floats('f32', [2], [1.0, 2.0]);
$norm = Toy\Tensor\rmsnorm($norm_x, $norm_w);
$norm_values = bytes_to_floats($norm->toBytes());
$inv_rms = 1.0 / sqrt(25.0 / 2.0 + 1e-6);
check('rmsnorm shape', $norm->shape() === [2]);
check_close('rmsnorm value 0', $norm_values[0], 3.0 * $inv_rms);
check_close('rmsnorm value 1', $norm_values[1], 8.0 * $inv_rms);

$norm_zero_eps = Toy\Tensor\rmsnorm($norm_x, $norm_w, 0.0);
$zero_eps_values = bytes_to_floats($norm_zero_eps->toBytes());
check_close('rmsnorm explicit eps', $zero_eps_values[0], 3.0 / sqrt(12.5));

check_throws('matmul rejects m that does not match', ValueError::class,
    fn () => Toy\Tensor\matmul_f32($a, $b, 3, 2, 2));
check_throws('matmul rejects n that does not match', ValueError::class,
    fn () => Toy\Tensor\matmul_f32($a, $b, 2, 3, 2));
check_throws('matmul rejects k that does not match', ValueError::class,
    fn () => Toy\Tensor\matmul_f32($a, $b, 2, 2, 3));
check_throws('matmul rejects a 1-D operand', ValueError::class,
    fn () => Toy\Tensor\matmul_f32($vector, $b, 2, 2, 2));
check_throws('matvec rejects a 2-D vector', ValueError::class,
    fn () => Toy\Tensor\matvec_f32($a, $b, 2, 2));
check_throws('add rejects different shapes', ValueError::class,
    fn () => Toy\Tensor\add($a, tensor_from_floats('f32', [2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])));
check_throws('add rejects different ranks', ValueError::class,
    fn () => Toy\Tensor\add($a, tensor_from_floats('f32', [4], [1.0, 2.0, 3.0, 4.0])));
check_throws('rmsnorm rejects a short weight', ValueError::class,
    fn () => Toy\Tensor\rmsnorm($norm_x, tensor_from_floats('f32', [1], [1.0])));
check_throws('rmsnorm rejects a 2-D input', ValueError::class,
    fn () => Toy\Tensor\rmsnorm($a, $norm_w));
check_throws('scale rejects a non-number factor', TypeError::class,
    fn () => Toy\Tensor\scale($a, []));
check_throws('functions reject foreign arguments', TypeError::class,
    fn () => Toy\Tensor\add($a, 'not a tensor'));

// The dtype property is the same for every factory and survives reshape
check('dtype survives reshape', $reshape->dtype === 'f32');
check('dtype survives operations', $matmul->dtype === 'f32');

if ($failures > 0) {
    fwrite(STDERR, "{$failures} of {$checks} checks failed\n");
    exit(1);
}

echo "all {$checks} checks passed\n";
