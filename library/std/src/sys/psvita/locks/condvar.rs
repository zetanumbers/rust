use crate::ptr::{self, NonNull};
use crate::sync::atomic::{self, AtomicPtr};
use crate::sys::locks::{mutex, Mutex};
use crate::time::Duration;

pub struct Condvar {
    inner: AtomicPtr<libc::cnd_impl_t>,
}

pub type MovableCondvar = Condvar;

impl Condvar {
    #[inline]
    pub const fn new() -> Condvar {
        Condvar { inner: AtomicPtr::new(ptr::null_mut()) }
    }

    #[inline]
    fn try_get(&self) -> Option<NonNull<libc::cnd_impl_t>> {
        let cond = self.inner.load(atomic::Ordering::Acquire);
        NonNull::new(cond)
    }

    #[inline]
    fn init_once(&self) -> NonNull<libc::cnd_impl_t> {
        self.try_get().unwrap_or_else(|| self.start_init())
    }

    #[cold]
    fn start_init(&self) -> NonNull<libc::cnd_impl_t> {
        let cond = Self::create_inner();
        match self.inner.compare_exchange(
            ptr::null_mut(),
            cond.as_ptr(),
            atomic::Ordering::AcqRel,
            atomic::Ordering::Acquire,
        ) {
            Ok(_) => cond,
            Err(older_mutex) => {
                let mut cond = cond.as_ptr();
                unsafe { libc::cnd_destroy(&mut cond) }
                NonNull::new(older_mutex).unwrap()
            }
        }
    }

    #[inline]
    fn create_inner() -> NonNull<libc::cnd_impl_t> {
        let mut cond = ptr::null_mut();
        assert_eq!(libc::thrd_success, unsafe { libc::cnd_init(&mut cond) });
        NonNull::new(cond).unwrap()
    }

    #[inline]
    pub unsafe fn notify_one(&self) {
        let mut cond = self.init_once().as_ptr();
        assert_eq!(libc::thrd_success, unsafe { libc::cnd_signal(&mut cond) })
    }

    #[inline]
    pub unsafe fn notify_all(&self) {
        let mut cond = self.init_once().as_ptr();
        assert_eq!(libc::thrd_success, unsafe { libc::cnd_broadcast(&mut cond) })
    }

    #[inline]
    pub unsafe fn wait(&self, mutex: &Mutex) {
        let mut cond = self.init_once().as_ptr();

        let mut mutex = mutex::raw(mutex);
        // mutex is locked, it cannot be null
        assert_eq!(libc::thrd_success, unsafe { libc::cnd_wait(&mut cond, &mut mutex) })
    }

    pub unsafe fn wait_timeout(&self, mutex: &Mutex, dur: Duration) -> bool {
        let mut xtime = libc::xtime { sec: 0, nsec: 0 };
        assert_eq!(libc::TIME_UTC, unsafe { libc::xtime_get(&mut xtime, libc::TIME_UTC) });

        let dur_nanos_before_next_sec = 1_000_000_000 - (dur.subsec_nanos() as i32);
        if dur_nanos_before_next_sec <= xtime.nsec {
            xtime.sec = xtime.sec.saturating_add(1);
            xtime.nsec -= dur_nanos_before_next_sec;
        } else {
            xtime.nsec += dur.subsec_nanos() as i32;
        }
        xtime.sec = xtime.sec.saturating_add(dur.as_secs().min(i32::MAX as u64) as i32);

        let mut cond = self.init_once().as_ptr();
        let mut mutex = mutex::raw(mutex);

        // mutex is locked, it cannot be null
        let err = unsafe { libc::cnd_timedwait(&mut cond, &mut mutex, &xtime) };
        match err {
            libc::thrd_success => true,
            libc::thrd_timeout => false,
            other => panic!("error code: {}", other),
        }
    }
}
