// When mutably indexing a type that implements `Index` but not `IndexMut`, a
// special 'help' message is added to the output.
// ... Except when it is not!
use std::borrow::Borrow;
use std::collections::HashMap;

#[derive(Hash, Eq, PartialEq)]
struct A {}
#[derive(Hash, Eq, PartialEq, Copy, Clone)]
struct ACopy {}
#[derive(Hash, Eq, PartialEq)]
struct B {}
#[derive(Hash, Eq, PartialEq)]
struct C {}
#[derive(Hash, Eq, PartialEq)]
struct D {}

impl Borrow<C> for D {
    fn borrow(&self) -> &C {
        &C {}
    }
}

/// In this test the key is a A type.
fn test_a_index() {
    let mut map = HashMap::<A, u32>::new();

    let index = &&&&&A {};
    // index gets autodereferenced
    map[index] = 23; //~ ERROR E0594
    map[&&&&&&&&index] = 23; //~ ERROR E0594
}
/// In this test the key is a A type. Auto-dereferencing to &A works but dereferencing to A need
/// the exact amount of `*`
fn test_a_2() {
    let mut map = HashMap::<A, u32>::new();

    let index = &&&&&A {};
    // complete dereferencing needs to be exact
    map.insert(*index, 23); //~ ERROR E0308
    // index gets autodereferenced
    if let Some(val) = map.get_mut(index) {
        *val = 23;
    } // passes
}
/// In this test the key is a A type. Could not be merged with 2 because compiler only shows error
/// 0308
fn test_a_3() {
    let mut map = HashMap::<A, u32>::new();

    let index = &&&&&A {};
    // A does not implement Copy so a Clone might be required
    map.insert(*****index, 23); //~ ERROR E0507
}

/// In this test the key is a ACopy type.
fn test_acopy_index() {
    let mut map = HashMap::<ACopy, u32>::new();

    let index = &&&&&ACopy {};
    // index gets autodereferenced
    map[index] = 23; //~ ERROR E0594
}
/// In this test the key is a ACopy type. Auto-dereferencing to &A works but dereferencing to A
/// need the exact amount of `*`.
fn test_acopy_2() {
    let mut map = HashMap::<ACopy, u32>::new();

    let index = &&&&&ACopy {};
    // complete dereferencing needs to be exact
    map.insert(*index, 23); //~ ERROR E0308

    // index gets autodereferenced
    if let Some(val) = map.get_mut(index) {
        *val = 23;
    } // passes
}
/// In this test the key is a ACopy type. Could not be merged with 2 because compiler only shows
/// error 0308
fn test_acopy_3() {
    let mut map = HashMap::<ACopy, u32>::new();

    let index = &&&&&ACopy {};
    map.insert(*****index, 23); // no E057 error in this case because ACopy is Copy
}

/// In this test the key type is B-reference. The autodereferencing does not work in this case for
/// both for the map[index] part and `get_mut` call.
/// This leads to E0277 errors.
fn test_b() {
    let mut map = HashMap::<&B, u32>::new();

    let index = &&&&&B {};
    // index does NOT get autorederefenced
    map[index] = 23; //~ ERROR E0277

    // index does NOT get autorederefenced
    if let Some(val) = map.get_mut(index) { //~ ERROR E0277
        *val = 23;
    }
    if let Some(val) = map.get_mut(***index) {
        *val = 23;
    } // passes
}


/// In this test the key type is C.
/// The `Borrow<C> for D` implementation changes nothing here, same error as for test_a.
fn test_c() {
    let mut map = HashMap::<C, u32>::new();

    let index = &&&&&C {};
    // index gets autodereferenced
    map[index] = 23; //~ ERROR E0594

    // index gets autodereferenced
    if let Some(val) = map.get_mut(index) {
        *val = 23;
    } // passes
}

/// In this test the key type is D. The `Borrow<C> for D` implementation seems to prevent
/// autodereferencing.
/// The autodereferencing does not work in this case for both for the map[index] part and `get_mut`
/// call.
/// This leads to E0277 errors.
fn test_d() {
    let mut map = HashMap::<D, u32>::new();

    let index = &&&&&D {};
    // index does NOT get autorederefenced
    map[index] = 23; //~ ERROR E0277

    // index does NOT get autorederefenced
    if let Some(val) = map.get_mut(index) {//~ ERROR E0277
        *val = 23;
    }
    if let Some(val) = map.get_mut(****index) {
        *val = 23;
    } // passes
}

fn main() {
    test_a_index();
    test_a_2();
    test_a_3();
    test_acopy_index();
    test_acopy_2();
    test_acopy_3();
    test_b();
    test_c();
    test_d();
}
