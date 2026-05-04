//! Regression test for https://github.com/rust-lang/rust/issues/31511

fn cast_thin_to_fat(x: *const ()) {
    x as *const [u8];
    //~^ ERROR: cannot cast thin pointer `*const ()` to wide pointer `*const [u8]`
}

fn main() {}
