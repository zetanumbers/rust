//@ build-fail

#![feature(unforgettable_types, forget_unsized)]

#[derive(Default)]
struct Unforgettable {
    _unforgettable: std::marker::PhantomUnforgettable,
}

fn main() {
    std::mem::forget(Unforgettable::default());
    std::mem::forget_unsized(Unforgettable::default());
}
