//! Regression test for hkl const closures not working in old solver

//@[next] check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

#![feature(const_trait_impl)]
#![feature(const_closures)]
const fn partial_compare() {
    let len_chain = const move |_a: &_, _b: &_| {};

    chaining_impl(len_chain);
    //[current]~^ ERROR: [const] FnOnce(&'a usize, &'a usize)` is not satisfied
}

const fn chaining_impl(x: impl for<'a> [const] FnOnce(&'a usize, &'a usize)) {
    std::mem::forget(x);
}

fn main() {}
