// Compiler:

extern "C" fn third(_a: usize, b: usize, c: usize) {
    let throw_away_f: fn((), usize, usize) =
        unsafe { std::mem::transmute(third as extern "C" fn(_, _, _)) };
    throw_away_f((), b, c)
}

fn main() {
    third(1, 2, 3);
}
