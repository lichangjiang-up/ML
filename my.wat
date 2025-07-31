;; (
;;     module
;;         (func $add (param $left i32) (param $right i32) (result i32)
;;         local.get $left
;;         local.get $right
;;         i32.add   
;;         )
;;     (export  "add" (func $add))
;; )

(module 
    (import "wasi_unstable" "fd_write" (func $fd_write (param i32 i32 i32 i32) (result i32)))
    (memory 1)
    (export "memory" (memory 0))
    (data (i32.const 8) "Hello \n")

    (func $main  (export "_start")
        (local $left i32)
        (local $right i32)
        (local $result i32)

        i32.const 1
        local.set $left

        i32.const 2
        local.set $right

        local.get $left
        local.get $right
        call $add
        local.set $result

        (i32.store (i32.const  0) (i32.const  8))
        (i32.store (i32.const  4) (i32.const  12))
        (call $fd_write 
            (i32.const 1)
            (i32.const 0)
            (i32.const 1)
            (i32.const 20)
        )
        drop
    )
    ;; (func $add_express (param $left i32) (param $right i32) (result ))
    (func $itoa (param $num i32) (result i32)
        local.get $num
        (i32.const 48)
        i32.add  ;; '0' 的 ASCII 码是 48
    )
    (func $add (param $lhs i32) (param $rhs i32) (result i32)
        local.get $lhs
        local.get $rhs
        i32.add
    )
    (export "add" (func $add))
    (export "itoa" (func $itoa))
)