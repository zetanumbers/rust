#[repr(u8)]
enum Priority { //~ HELP: add `#[derive(Copy, Clone)]` to the enum definition
    High = 255,
    Normal = 127,
    Low = 1,
}

fn main() {
    let priority = &Priority::Normal;
    let priority = priority as u8; //~ ERROR casting `&Priority` as `u8` is invalid
    //~| HELP: try dereferencing before the cast

    let priority = &Priority::Normal as u8; //~ ERROR casting `&Priority` as `u8` is invalid
    //~| HELP: cast through a raw pointer first
}
