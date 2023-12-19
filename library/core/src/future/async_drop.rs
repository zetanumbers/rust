#![unstable(feature = "async_drop", issue = "none")]

use crate::pin::Pin;
use crate::task::{Context, Poll};

#[unstable(feature = "async_drop", issue = "none")]
#[lang = "async_drop"]
pub trait AsyncDrop {
    #[unstable(feature = "async_drop", issue = "none")]
    #[lang = "poll_drop"]
    fn poll_drop(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()>;
}
