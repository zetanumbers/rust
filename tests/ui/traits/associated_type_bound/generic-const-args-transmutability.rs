//! Regression test for <https://github.com/rust-lang/rust/issues/156293>
//@ edition: 2024
//@ check-pass

#![feature(min_generic_const_args, transmutability)]

trait Foo {
    type AssocB: std::mem::TransmuteFrom<()>;
}

fn main() {}
