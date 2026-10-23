--TEST--
Toy\Tensor validation errors and protected object state
--EXTENSIONS--
toy
--FILE--
<?php
function show(callable $fn): void {
    try {
        $fn();
        echo "no exception\n";
    } catch (Throwable $e) {
        echo get_class($e), "\n";
    }
}
$t = Toy\Tensor::zeros('f32', [2, 2]);
show(fn() => Toy\Tensor::zeros('q8_0', [32]));
show(fn() => Toy\Tensor::zeros('f64', [1]));
show(fn() => Toy\Tensor::zeros('f32', []));
show(fn() => Toy\Tensor::zeros('f32', [0]));
show(fn() => Toy\Tensor::zeros('f32', [1.5]));
show(fn() => Toy\Tensor::zeros('f32', [1 => 1, 0 => 2]));
show(fn() => Toy\Tensor::fromBytes('f32', [2, 2], str_repeat("\0", 15)));
show(fn() => $t->reshape([5]));
show(fn() => $t->cast('q8_0'));
show(function () use ($t): void { $t->dtype = 'q8_0'; });
show(fn() => new Toy\Tensor());
show(fn() => clone $t);
show(function () use ($t): void { $t->extra = 1; });
show(fn() => serialize($t));
show(fn() => Toy\Tensor\scale($t, []));
show(fn() => Toy\Tensor\add($t, 'x'));
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
Error
Error
Error
Error
Exception
TypeError
TypeError
done
