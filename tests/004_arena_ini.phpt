--TEST--
toy.kernel_arena_size INI directive and module info
--EXTENSIONS--
toy
--FILE--
<?php
var_dump(ini_get('toy.kernel_arena_size'));
var_dump(ini_set('toy.kernel_arena_size', '16M'));
var_dump(ini_get('toy.kernel_arena_size'));

ob_start();
phpinfo(INFO_MODULES);
$info = ob_get_clean();
var_dump(str_contains($info, 'kernel arena'));
var_dump(str_contains($info, 'kernel spec cache'));
var_dump(str_contains($info, 'toy.kernel_arena_size'));
?>
--EXPECT--
string(3) "32M"
bool(false)
string(3) "32M"
bool(true)
bool(true)
bool(true)
