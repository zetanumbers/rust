// Regression test for issue #156264

mod m_pub {
    pub struct S {}
}

mod m_crate {
    pub(crate) use crate::m_pub::S;
}

pub(crate) use m_crate::*;
//~^ ERROR `S` is only public within the crate, and cannot be re-exported outside
pub use m_pub::*;

fn main() {}
