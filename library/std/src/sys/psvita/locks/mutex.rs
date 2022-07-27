use crate::ptr::{self, NonNull};
use crate::sync::atomic::{self, AtomicPtr};

pub type MovableMutex = Mutex;

pub struct Mutex {
    inner: AtomicPtr<libc::mtx_impl_t>,
}

#[inline]
pub fn raw(mutex: &Mutex) -> libc::mtx_t {
    mutex.inner.load(atomic::Ordering::Acquire)
}

impl Mutex {
    #[inline]
    pub const fn new() -> Mutex {
        Mutex { inner: AtomicPtr::new(ptr::null_mut()) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
        *self.inner.get_mut() = Self::create_inner().as_ptr();
    }

    #[inline]
    fn try_get(&self) -> Option<NonNull<libc::mtx_impl_t>> {
        let mutex = self.inner.load(atomic::Ordering::Acquire);
        NonNull::new(mutex)
    }

    #[inline]
    fn init_once(&self) -> NonNull<libc::mtx_impl_t> {
        self.try_get().unwrap_or_else(|| self.start_init())
    }

    #[cold]
    fn start_init(&self) -> NonNull<libc::mtx_impl_t> {
        let mutex = Self::create_inner();
        match self.inner.compare_exchange(
            ptr::null_mut(),
            mutex.as_ptr(),
            atomic::Ordering::AcqRel,
            atomic::Ordering::Acquire,
        ) {
            Ok(_) => mutex,
            Err(older_mutex) => {
                let mut mutex = mutex.as_ptr();
                unsafe { libc::mtx_destroy(&mut mutex) }
                NonNull::new(older_mutex).unwrap()
            }
        }
    }

    #[inline]
    fn create_inner() -> NonNull<libc::mtx_impl_t> {
        let mut mutex = ptr::null_mut();
        assert_eq!(libc::thrd_success, unsafe { libc::mtx_init(&mut mutex, libc::mtx_plain) });
        NonNull::new(mutex).unwrap()
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let mut mutex = self.init_once().as_ptr();
        assert_eq!(
            libc::thrd_success,
            unsafe { libc::mtx_lock(&mut mutex) },
            "Perhaps current thread already owns the lock to the mutex"
        );
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        if let Some(mutex) = self.try_get() {
            let mut mutex = mutex.as_ptr();
            assert_eq!(libc::thrd_success, unsafe { libc::mtx_unlock(&mut mutex) });
        } else {
            panic!("Trying to unlock an uninitialized Mutex");
        }
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let mut mutex = self.init_once().as_ptr();
        libc::thrd_success == unsafe { libc::mtx_trylock(&mut mutex) }
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        // Performs the null check internally
        unsafe { libc::mtx_destroy(self.inner.get_mut()) }
    }
}
