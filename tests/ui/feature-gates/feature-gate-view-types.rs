//@ known-bug: #155938

struct Foo {
    a: usize,
    b: usize,
}

fn bar(a: &mut Foo.{ a }, b: &mut Foo.{ b }) {
    a.a += 1;
    b.b += 1;
}

fn main() {
    let mut foo = Foo { a: 0, b: 0 };
    bar(&mut foo, &mut foo);
}
