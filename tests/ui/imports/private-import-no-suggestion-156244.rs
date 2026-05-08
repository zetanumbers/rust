//@ edition: 2018

mod a {
    pub struct One;
    pub struct Two;
}

mod b {
    use crate::a::One;
    use crate::a::Two;
}

mod outer {
    pub mod actual {
        pub struct Item;
    }
}

mod rename {
    use crate::outer::actual as inner;
}

mod bad {
    use crate::b::{One, Two};
    //~^ ERROR struct import `One` is private [E0603]
    //~| ERROR struct import `Two` is private [E0603]
    use crate::rename::inner::Item as Item1;
    //~^ ERROR module import `inner` is private [E0603]
}

fn main() {}
