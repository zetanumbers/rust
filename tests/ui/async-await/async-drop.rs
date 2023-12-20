// run-pass
// check-run-results

#![feature(async_drop)]
#![allow(incomplete_features)]

// edition: 2021

use std::{
    future::{AsyncDrop, Future},
    pin::{pin, Pin},
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
};

fn main() {
    {
        let _ = Foo::new(0);
    }
    let bar = bar(1);
    println!("after bar(1), before block_on(bar)");
    block_on(bar);
    println!("done")
}

// Uses 1 ident
async fn bar(ident_base: usize) {
    let mut _first = Foo::new(ident_base);
}

#[derive(Debug)]
struct Foo {
    ident: usize,
    stage: FooStage,
}

impl Foo {
    fn new(ident: usize) -> Self {
        let out = Foo {
            ident,
            stage: FooStage::Init,
        };
        println!("Foo::new() -> {out:?}");
        out
    }
}

impl Foo {
    fn inner_poll(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        caller: &'static str,
    ) -> Poll<usize> {
        println!("Foo::{}({:?})", caller, *self);
        match self.stage {
            FooStage::Init => {
                self.stage = FooStage::Yield0;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            FooStage::Yield0 => {
                self.stage = FooStage::Done;
                cx.waker().wake_by_ref();
                Poll::Ready(self.ident)
            }
            FooStage::Done => panic!("AsyncDrop of Foo is Done!"),
        }
    }
}

impl AsyncDrop for Foo {
    fn poll_drop(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        self.inner_poll(cx, "poll_drop").map(|_| ())
    }
}

impl Drop for Foo {
    fn drop(&mut self) {
        println!("Foo::drop({self:?})");
    }
}

#[derive(Debug, Clone, Copy)]
enum FooStage {
    Init,
    Yield0,
    Done,
}

fn block_on<F>(fut: F) -> F::Output
where
    F: Future,
{
    let mut fut = pin!(fut);
    let (waker, rx) = simple_waker();
    let mut context = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut context) {
            Poll::Ready(out) => break out,
            // expect wake in polls
            Poll::Pending => rx.try_recv().unwrap(),
        }
    }
}

fn simple_waker() -> (Waker, mpsc::Receiver<()>) {
    struct SimpleWaker {
        tx: std::sync::mpsc::Sender<()>,
    }

    impl Wake for SimpleWaker {
        fn wake(self: Arc<Self>) {
            self.tx.send(()).unwrap();
        }
    }

    let (tx, rx) = mpsc::channel();
    (Waker::from(Arc::new(SimpleWaker { tx })), rx)
}
