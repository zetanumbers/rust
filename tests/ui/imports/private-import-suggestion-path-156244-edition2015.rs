//@ edition: 2015

mod one {
    pub struct One();
}

mod two {
    use one::One;
}

mod test {
    use two::One;
    //~^ ERROR struct import `One` is private [E0603]
}

fn main() {}
