//@ build-fail

#![feature(unforgettable_types)]

#[derive(Debug, Default)]
struct Unforgettable {
    _unforgettable: std::marker::PhantomUnforgettable,
}

fn main() {
    let _: &dyn Send = &Unforgettable::default();
    //~^ ERROR `PhantomUnforgettable` cannot be forgotten
}
