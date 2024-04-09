use super::marker_impls;

/// A marker for types that can be forgotten.
#[unstable(feature = "unforgettable_types", issue = "none")]
#[cfg_attr(not(test), rustc_diagnostic_item = "Forget")]
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot be forgotten",
    label = "`{Self}` cannot be forgotten"
)]
#[lang = "forget"]
pub unsafe auto trait Forget {
    // empty.
}

marker_impls! {
    #[unstable(feature = "unforgettable_types", issue = "none")]
    unsafe Forget for
        {T: ?Sized} &T,
        {T: ?Sized} &mut T,
}

marker_impls! {
    #[unstable(feature = "unforgettable_types", issue = "none")]
    unsafe Forget for
        {T: ?Sized} *const T,
        {T: ?Sized} *mut T,
}

/// A marker type which does not implement `Forget`.
///
/// If a type contains a `PhantomUnforgettable`, it will not implement `Forget` by default.
// TODO: use `unsafe impl<T: 'static> Forget for Unforgettable<T> {}` instead
#[unstable(feature = "unforgettable_types", issue = "none")]
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PhantomUnforgettable;

#[unstable(feature = "unforgettable_types", issue = "none")]
impl !Forget for PhantomUnforgettable {}
