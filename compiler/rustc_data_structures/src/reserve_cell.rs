use std::cell::{Cell, RefCell, RefMut};
use std::ptr::NonNull;

#[derive(Default)]
pub struct ReserveCell<T> {
    next: Cell<Option<NonNull<ReserveCellData<T>>>>,
}

#[derive(Default)]
struct ReserveCellData<T> {
    data: RefCell<T>,
    inner: ReserveCell<T>,
}

unsafe impl<T: Send> Send for ReserveCell<T> {}

impl<T: Default> ReserveCell<T> {
    pub fn reserve(&self) -> RefMut<'_, T> {
        if let Some(mut ptr) = self.next.get() {
            let res = unsafe { ptr.as_mut().data.try_borrow_mut() };
            if let Ok(guard) = res {
                guard
            } else {
                // SAFETY: (newly initialized) self.next pointer is
                // valid until ReserveCell's drop
                unsafe { ptr.as_ref().inner.reserve() }
            }
        } else {
            // SAFETY: Box::into_raw only returns non-null pointers
            let mut ptr = unsafe { NonNull::new_unchecked(Box::into_raw(Box::default())) };

            self.next.set(Some(ptr));

            // SAFETY: ptr was initialized just now
            unsafe { ptr.as_mut().data.borrow_mut() }
        }
    }
}

impl<T> ReserveCell<T> {
    pub const fn new() -> Self {
        ReserveCell { next: Cell::new(None) }
    }

    pub fn pop_first(&self) -> Option<T> {
        self.next.get().map(|ptr| {
            // SAFETY: self.next pointer, if already non-null, comes
            // from Box::into_raw and stays valid until this moment.
            let cell_data = unsafe { Box::from_raw(ptr.as_ptr()) };

            // Moves next cell to self
            self.next.set(cell_data.inner.next.take());
            cell_data.data.into_inner()
        })
    }
}

impl<T> IntoIterator for ReserveCell<T> {
    type Item = T;

    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { cell: self }
    }
}

pub struct IntoIter<T> {
    cell: ReserveCell<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.cell.pop_first()
    }
}

impl<T> Drop for ReserveCell<T> {
    fn drop(&mut self) {
        if let Some(ptr) = *self.next.get_mut() {
            // SAFETY: self.next pointer, if already non-null, comes
            // from Box::into_raw and stays valid until this moment
            unsafe { drop(Box::from_raw(ptr.as_ptr())) };
        }
    }
}
