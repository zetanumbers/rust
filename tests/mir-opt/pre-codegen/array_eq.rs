//@ compile-flags: -O -Zmir-opt-level=2
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY

#![crate_type = "lib"]

// EMIT_MIR array_eq.eq_odd_length.PreCodegen.after.mir
pub unsafe fn eq_odd_length<T: Copy>(a: &[u8; 3], b: &[u8; 3]) -> bool {
    // CHECK-LABEL: fn eq_odd_length(_1: &[u8; 3], _2: &[u8; 3]) -> bool
    // CHECK: _0 = raw_eq::<[u8; 3]>(move _1, move _2)
    a == b
}

// EMIT_MIR array_eq.eq_ipv4.PreCodegen.after.mir
pub unsafe fn eq_ipv4<T: Copy>(a: &[u8; 4], b: &[u8; 4]) -> bool {
    // CHECK-LABEL: fn eq_ipv4(_1: &[u8; 4], _2: &[u8; 4]) -> bool
    // CHECK: _0 = raw_eq::<[u8; 4]>(move _1, move _2)
    a == b
}

// EMIT_MIR array_eq.eq_ipv6.PreCodegen.after.mir
pub unsafe fn eq_ipv6<T: Copy>(a: &[u16; 8], b: &[u16; 8]) -> bool {
    // CHECK-LABEL: fn eq_ipv6(_1: &[u16; 8], _2: &[u16; 8]) -> bool
    // CHECK: _0 = raw_eq::<[u16; 8]>(move _1, move _2)
    a == b
}
