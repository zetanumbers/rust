// PR #156244 comment

mod one {
    pub struct One();
}

mod two {
    use crate::one::One;
}

mod test {
    use crate::two::One;
    //~^ ERROR struct import `One` is private [E0603]
}

fn main() {}
