use crate::alloc::{GlobalAlloc, Layout, System};
use crate::ptr;
use crate::sys::common::alloc::MIN_ALIGN;

#[stable(feature = "alloc_system_type", since = "1.28.0")]
unsafe impl GlobalAlloc for System {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            unsafe { libc::malloc(layout.size()) as *mut u8 }
        } else {
            unsafe { libc::memalign(layout.align(), layout.size()) as *mut u8 }
        }
    }

    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= layout.size() {
            unsafe { libc::calloc(layout.size(), 1) as *mut u8 }
        } else {
            let ptr = unsafe { self.alloc(layout) };
            if !ptr.is_null() {
                unsafe {
                    ptr::write_bytes(ptr, 0, layout.size());
                }
            }
            ptr
        }
    }

    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { libc::free(ptr as *mut libc::c_void) }
    }

    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.align() <= MIN_ALIGN && layout.align() <= new_size {
            unsafe { libc::realloc(ptr as *mut libc::c_void, new_size) as *mut u8 }
        } else {
            unsafe {
                libc::reallocalign(ptr as *mut libc::c_void, new_size, layout.align()) as *mut u8
            }
        }
    }
}

#[doc(hidden)]
#[no_mangle]
#[link_section = ".rodata.SceModuleInfo"]
#[linkage = "weak"]
pub static sceLibcHeapSize: usize = usize::MAX;

#[doc(hidden)]
#[no_mangle]
#[link_section = ".rodata.SceModuleInfo"]
#[linkage = "weak"]
pub static sceLibcHeapExtendedAlloc: libc::SceBool = libc::SCE_TRUE;
