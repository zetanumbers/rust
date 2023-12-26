// run-pass
// check-run-results

// compile-flags: --edition 2024 -Z unstable-options

#![feature(defer_blocks)]
#![allow(incomplete_features)]

fn main() {
    defer {
        println!("hello");
    }
    println!("world");
}
