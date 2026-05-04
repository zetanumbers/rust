//! Regression test for https://github.com/rust-lang/rust/issues/30615

//@ run-pass
fn main() {
    &0u8 as *const u8 as *const dyn PartialEq<u8>;
    &[0u8] as *const [u8; 1] as *const [u8];
}
