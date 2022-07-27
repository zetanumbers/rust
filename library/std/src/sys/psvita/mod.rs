#![deny(unsafe_op_in_unsafe_fn)]

pub mod alloc;
pub mod args;
#[path = "../unix/cmath.rs"]
pub mod cmath;
pub mod env;
pub mod fs;
pub mod io;
pub mod locks;
pub mod net;
pub mod os;
#[path = "../unix/os_str.rs"]
pub mod os_str;
#[path = "../unix/path.rs"]
pub mod path;
pub mod pipe;
pub mod process;
pub mod stdio;
pub mod thread;
#[cfg(target_thread_local)]
pub mod thread_local_dtor;
pub mod thread_local_key;
pub mod time;

mod common;
pub use common::*;

pub type UID = crate::num::NonZeroI32;
pub type AtomicUID = crate::sync::atomic::AtomicI32;

pub fn cvt_uid(ret: libc::c_int) -> crate::io::Result<Option<UID>> {
    if ret >= 0 { Ok(UID::new(ret)) } else { Err(crate::io::Error::from_raw_os_error(ret)) }
}

pub fn cvt_nz(ret: libc::c_int) -> crate::io::Result<()> {
    if ret == 0 { Ok(()) } else { Err(crate::io::Error::from_raw_os_error(ret)) }
}
