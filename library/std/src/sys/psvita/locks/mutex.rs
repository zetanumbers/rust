use crate::ptr;
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
        *self.inner.get_mut() = Self::create_inner();
    }

    #[inline]
    fn init_once(&self) -> libc::mtx_t {
        let mutex = self.inner.load(atomic::Ordering::Acquire);
        if mutex.is_null() { self.start_init() } else { mutex }
    }

    #[cold]
    fn start_init(&self) -> libc::mtx_t {
        let mut mutex = Self::create_inner();
        match self.inner.compare_exchange(
            ptr::null_mut(),
            mutex,
            atomic::Ordering::AcqRel,
            atomic::Ordering::Acquire,
        ) {
            Ok(_) => mutex,
            Err(older_mutex) => {
                unsafe { libc::mtx_destroy(&mut mutex) }
                older_mutex
            }
        }
    }

    #[inline]
    fn create_inner() -> libc::mtx_t {
        let mut mutex = ptr::null_mut();
        assert_eq!(libc::thrd_success, unsafe { libc::mtx_init(&mut mutex, libc::mtx_plain) });
        mutex
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let mut mutex = self.init_once();
        assert_eq!(
            libc::thrd_success,
            unsafe { libc::mtx_lock(&mut mutex) },
            "Perhaps current thread already owns the lock to the mutex"
        );
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        let mut mutex = self.init_once();
        assert_eq!(libc::thrd_success, unsafe { libc::mtx_unlock(&mut mutex) });
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let mut mutex = self.init_once();
        match unsafe { libc::mtx_trylock(&mut mutex) } {
            libc::thrd_success => true,
            libc::thrd_busy => false,
            other => panic!("error code: {}", other),
        }
    }
}

impl Drop for Mutex {
    fn drop(&mut self) {
        // Performs the null check internally
        unsafe { libc::mtx_destroy(self.inner.get_mut()) }
    }
}
