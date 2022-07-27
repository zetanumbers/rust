use crate::sys::{cvt_nz, cvt_uid, AtomicUID, UID};
use crate::{ptr, sync::atomic};

pub struct RwLock {
    uid: AtomicUID,
}

pub type MovableRwLock = RwLock;

impl RwLock {
    #[inline]
    pub const fn new() -> RwLock {
        RwLock { uid: AtomicUID::new(0) }
    }

    #[inline]
    fn init_once(&self) -> UID {
        self.try_get().unwrap_or_else(|| self.start_init())
    }

    #[inline]
    fn try_get(&self) -> Option<UID> {
        let rwlock = self.uid.load(atomic::Ordering::Acquire);
        UID::new(rwlock)
    }

    #[cold]
    fn start_init(&self) -> UID {
        let rwlock = Self::create_inner();
        match self.uid.compare_exchange(
            0,
            rwlock.get(),
            atomic::Ordering::AcqRel,
            atomic::Ordering::Acquire,
        ) {
            Ok(_) => rwlock,
            Err(older) => {
                cvt_nz(unsafe { libc::sceKernelDeleteRWLock(rwlock.get()) }).unwrap();
                UID::new(older).unwrap()
            }
        }
    }

    #[inline]
    fn create_inner() -> UID {
        const NAME: &str = "rust/std\0";
        cvt_uid(unsafe { libc::sceKernelCreateRWLock(NAME.as_ptr().cast(), 0, ptr::null()) })
            .unwrap()
            .unwrap()
    }

    #[inline]
    pub unsafe fn read(&self) {
        let uid = self.init_once();
        cvt_nz(unsafe { libc::sceKernelLockReadRWLock(uid.get(), ptr::null_mut()) }).unwrap();
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let uid = self.init_once();
        0 == unsafe { libc::sceKernelTryLockReadRWLock(uid.get()) }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let uid = self.init_once();
        cvt_nz(unsafe { libc::sceKernelLockWriteRWLock(uid.get(), ptr::null_mut()) }).unwrap();
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let uid = self.init_once();
        0 == unsafe { libc::sceKernelTryLockWriteRWLock(uid.get()) }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        if let Some(uid) = self.try_get() {
            cvt_nz(unsafe { libc::sceKernelUnlockReadRWLock(uid.get()) }).unwrap();
        } else {
            panic!("Trying to read_unlock uninitialized RwLock");
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        if let Some(uid) = self.try_get() {
            cvt_nz(unsafe { libc::sceKernelUnlockWriteRWLock(uid.get()) }).unwrap();
        } else {
            panic!("Trying to write_unlock uninitialized RwLock");
        }
    }
}
