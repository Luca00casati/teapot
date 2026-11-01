//! The kernel interfaces, declared by hand (no libc or libdrm crate): the few
//! libc calls std does not wrap, and the DRM/KMS and i915 uapi structures and
//! ioctl numbers from <drm/drm.h>, <drm/drm_mode.h> and <drm/i915_drm.h>.

use std::ffi::c_void;
use std::mem::size_of;
use std::os::fd::RawFd;

mod ffi {
    use std::ffi::c_void;
    extern "C" {
        pub fn ioctl(fd: i32, req: u64, ...) -> i32;
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn poll(fds: *mut super::PollFd, n: u64, timeout_ms: i32) -> i32;
        pub fn read(fd: i32, buf: *mut c_void, n: usize) -> isize;
        pub fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
}

pub const SIGINT: i32 = 2;
pub const SIGTERM: i32 = 15;

#[repr(C)]
pub struct PollFd {
    fd: i32,
    events: i16,
    revents: i16,
}

/// true when the ioctl succeeded
pub fn ioctl<T>(fd: RawFd, req: u64, arg: &mut T) -> bool {
    unsafe { ffi::ioctl(fd, req, arg as *mut T) == 0 }
}
pub fn ioctl_none(fd: RawFd, req: u64) -> bool {
    unsafe { ffi::ioctl(fd, req, 0usize) == 0 }
}

/// a shared read/write mapping of `len` bytes of `fd` at `off`
pub fn mmap(fd: RawFd, len: u64, off: u64) -> Option<*mut u8> {
    const PROT_RW: i32 = 3;
    const MAP_SHARED: i32 = 1;
    let p = unsafe { ffi::mmap(std::ptr::null_mut(), len as usize, PROT_RW, MAP_SHARED, fd, off as i64) };
    if p as isize == -1 { None } else { Some(p as *mut u8) }
}
pub fn munmap(p: *mut u8, len: u64) {
    unsafe { ffi::munmap(p as *mut c_void, len as usize) };
}

/// does `fd` become readable within `timeout_ms` (-1 = forever)?
pub fn poll_in(fd: RawFd, timeout_ms: i32) -> bool {
    const POLLIN: i16 = 1;
    let mut p = PollFd { fd, events: POLLIN, revents: 0 };
    unsafe { ffi::poll(&mut p, 1, timeout_ms) > 0 }
}
/// raw read(2), bypassing std's buffering (the bytes only signal a key press)
pub fn read(fd: RawFd, buf: &mut [u8]) -> isize {
    unsafe { ffi::read(fd, buf.as_mut_ptr() as *mut c_void, buf.len()) }
}
pub fn signal(sig: i32, handler: extern "C" fn(i32)) {
    unsafe { ffi::signal(sig, handler) };
}

pub fn perror(what: &str) {
    eprintln!("{what}: {}", std::io::Error::last_os_error());
}

/* ------------------------------------------------------ ioctl numbers --- */

const fn ioc(dir: u64, nr: u64, size: usize) -> u64 {
    dir << 30 | (size as u64) << 16 | (b'd' as u64) << 8 | nr
}
const fn iowr<T>(nr: u64) -> u64 {
    ioc(3, nr, size_of::<T>())
}
const fn iow<T>(nr: u64) -> u64 {
    ioc(1, nr, size_of::<T>())
}
const DRM_COMMAND_BASE: u64 = 0x40;

pub const DRM_IOCTL_GEM_CLOSE: u64 = iow::<DrmGemClose>(0x09);
pub const DRM_IOCTL_SET_MASTER: u64 = ioc(0, 0x1e, 0);
pub const DRM_IOCTL_MODE_GETRESOURCES: u64 = iowr::<DrmModeCardRes>(0xA0);
pub const DRM_IOCTL_MODE_GETCRTC: u64 = iowr::<DrmModeCrtc>(0xA1);
pub const DRM_IOCTL_MODE_SETCRTC: u64 = iowr::<DrmModeCrtc>(0xA2);
pub const DRM_IOCTL_MODE_GETENCODER: u64 = iowr::<DrmModeGetEncoder>(0xA6);
pub const DRM_IOCTL_MODE_GETCONNECTOR: u64 = iowr::<DrmModeGetConnector>(0xA7);
pub const DRM_IOCTL_MODE_ADDFB: u64 = iowr::<DrmModeFbCmd>(0xAE);
pub const DRM_IOCTL_MODE_RMFB: u64 = iowr::<u32>(0xAF);
pub const DRM_IOCTL_MODE_PAGE_FLIP: u64 = iowr::<DrmModeCrtcPageFlip>(0xB0);
pub const DRM_IOCTL_MODE_DIRTYFB: u64 = iowr::<DrmModeFbDirtyCmd>(0xB1);
pub const DRM_IOCTL_MODE_CREATE_DUMB: u64 = iowr::<DrmModeCreateDumb>(0xB2);
pub const DRM_IOCTL_MODE_MAP_DUMB: u64 = iowr::<DrmModeMapDumb>(0xB3);
pub const DRM_IOCTL_MODE_DESTROY_DUMB: u64 = iowr::<DrmModeDestroyDumb>(0xB4);
pub const DRM_IOCTL_I915_GETPARAM: u64 = iowr::<DrmI915Getparam>(DRM_COMMAND_BASE + 0x06);
pub const DRM_IOCTL_I915_GEM_CREATE: u64 = iowr::<DrmI915GemCreate>(DRM_COMMAND_BASE + 0x1b);
pub const DRM_IOCTL_I915_GEM_SET_DOMAIN: u64 = iow::<DrmI915GemSetDomain>(DRM_COMMAND_BASE + 0x1f);
pub const DRM_IOCTL_I915_GEM_MMAP_OFFSET: u64 = iowr::<DrmI915GemMmapOffset>(DRM_COMMAND_BASE + 0x24);
pub const DRM_IOCTL_I915_GEM_EXECBUFFER2: u64 = iow::<DrmI915GemExecbuffer2>(DRM_COMMAND_BASE + 0x29);
pub const DRM_IOCTL_I915_GET_RESET_STATS: u64 = iowr::<DrmI915ResetStats>(DRM_COMMAND_BASE + 0x32);

pub const DRM_MODE_TYPE_PREFERRED: u32 = 1 << 3;
pub const DRM_MODE_PAGE_FLIP_EVENT: u32 = 1;
pub const I915_PARAM_CHIPSET_ID: i32 = 4;
pub const I915_MMAP_OFFSET_WB: u64 = 2;
pub const I915_GEM_DOMAIN_CPU: u32 = 1;
pub const I915_GEM_DOMAIN_RENDER: u32 = 2;
pub const EXEC_OBJECT_WRITE: u64 = 1 << 2;
pub const I915_EXEC_RENDER: u64 = 1;

/* --------------------------------------------------------- structures --- */

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeCardRes {
    pub fb_id_ptr: u64,
    pub crtc_id_ptr: u64,
    pub connector_id_ptr: u64,
    pub encoder_id_ptr: u64,
    pub count_fbs: u32,
    pub count_crtcs: u32,
    pub count_connectors: u32,
    pub count_encoders: u32,
    pub min_width: u32,
    pub max_width: u32,
    pub min_height: u32,
    pub max_height: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeModeinfo {
    pub clock: u32,
    pub hdisplay: u16,
    pub hsync_start: u16,
    pub hsync_end: u16,
    pub htotal: u16,
    pub hskew: u16,
    pub vdisplay: u16,
    pub vsync_start: u16,
    pub vsync_end: u16,
    pub vtotal: u16,
    pub vscan: u16,
    pub vrefresh: u32,
    pub flags: u32,
    pub type_: u32,
    pub name: [u8; 32],
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeCrtc {
    pub set_connectors_ptr: u64,
    pub count_connectors: u32,
    pub crtc_id: u32,
    pub fb_id: u32,
    pub x: u32,
    pub y: u32,
    pub gamma_size: u32,
    pub mode_valid: u32,
    pub mode: DrmModeModeinfo,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeGetEncoder {
    pub encoder_id: u32,
    pub encoder_type: u32,
    pub crtc_id: u32,
    pub possible_crtcs: u32,
    pub possible_clones: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeGetConnector {
    pub encoders_ptr: u64,
    pub modes_ptr: u64,
    pub props_ptr: u64,
    pub prop_values_ptr: u64,
    pub count_modes: u32,
    pub count_props: u32,
    pub count_encoders: u32,
    pub encoder_id: u32,
    pub connector_id: u32,
    pub connector_type: u32,
    pub connector_type_id: u32,
    pub connection: u32,
    pub mm_width: u32,
    pub mm_height: u32,
    pub subpixel: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeFbCmd {
    pub fb_id: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub bpp: u32,
    pub depth: u32,
    pub handle: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeCrtcPageFlip {
    pub crtc_id: u32,
    pub fb_id: u32,
    pub flags: u32,
    pub reserved: u32,
    pub user_data: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeFbDirtyCmd {
    pub fb_id: u32,
    pub flags: u32,
    pub color: u32,
    pub num_clips: u32,
    pub clips_ptr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeCreateDumb {
    pub height: u32,
    pub width: u32,
    pub bpp: u32,
    pub flags: u32,
    pub handle: u32,
    pub pitch: u32,
    pub size: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeMapDumb {
    pub handle: u32,
    pub pad: u32,
    pub offset: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmModeDestroyDumb {
    pub handle: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmGemClose {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub struct DrmI915Getparam {
    pub param: i32,
    pub value: *mut i32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemCreate {
    pub size: u64,
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemMmapOffset {
    pub handle: u32,
    pub pad: u32,
    pub offset: u64,
    pub flags: u64,
    pub extensions: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemSetDomain {
    pub handle: u32,
    pub read_domains: u32,
    pub write_domain: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemRelocationEntry {
    pub target_handle: u32,
    pub delta: u32,
    pub offset: u64,
    pub presumed_offset: u64,
    pub read_domains: u32,
    pub write_domain: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemExecObject2 {
    pub handle: u32,
    pub relocation_count: u32,
    pub relocs_ptr: u64,
    pub alignment: u64,
    pub offset: u64,
    pub flags: u64,
    pub rsvd1: u64,
    pub rsvd2: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915GemExecbuffer2 {
    pub buffers_ptr: u64,
    pub buffer_count: u32,
    pub batch_start_offset: u32,
    pub batch_len: u32,
    pub dr1: u32,
    pub dr4: u32,
    pub num_cliprects: u32,
    pub cliprects_ptr: u64,
    pub flags: u64,
    pub rsvd1: u64,
    pub rsvd2: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct DrmI915ResetStats {
    pub ctx_id: u32,
    pub flags: u32,
    pub reset_count: u32,
    pub batch_active: u32,
    pub batch_pending: u32,
    pub pad: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    // values printed by a C program built against the system's uapi headers
    #[test]
    fn uapi_matches_headers() {
        assert_eq!(DRM_IOCTL_SET_MASTER, 0x641e);
        assert_eq!(DRM_IOCTL_MODE_GETRESOURCES, 0xc04064a0);
        assert_eq!(DRM_IOCTL_MODE_GETCRTC, 0xc06864a1);
        assert_eq!(DRM_IOCTL_MODE_SETCRTC, 0xc06864a2);
        assert_eq!(DRM_IOCTL_MODE_GETENCODER, 0xc01464a6);
        assert_eq!(DRM_IOCTL_MODE_GETCONNECTOR, 0xc05064a7);
        assert_eq!(DRM_IOCTL_MODE_ADDFB, 0xc01c64ae);
        assert_eq!(DRM_IOCTL_MODE_RMFB, 0xc00464af);
        assert_eq!(DRM_IOCTL_MODE_PAGE_FLIP, 0xc01864b0);
        assert_eq!(DRM_IOCTL_MODE_DIRTYFB, 0xc01864b1);
        assert_eq!(DRM_IOCTL_MODE_CREATE_DUMB, 0xc02064b2);
        assert_eq!(DRM_IOCTL_MODE_MAP_DUMB, 0xc01064b3);
        assert_eq!(DRM_IOCTL_MODE_DESTROY_DUMB, 0xc00464b4);
        assert_eq!(DRM_IOCTL_GEM_CLOSE, 0x40086409);
        assert_eq!(DRM_IOCTL_I915_GETPARAM, 0xc0106446);
        assert_eq!(DRM_IOCTL_I915_GEM_CREATE, 0xc010645b);
        assert_eq!(DRM_IOCTL_I915_GEM_MMAP_OFFSET, 0xc0206464);
        assert_eq!(DRM_IOCTL_I915_GEM_SET_DOMAIN, 0x400c645f);
        assert_eq!(DRM_IOCTL_I915_GEM_EXECBUFFER2, 0x40406469);
        assert_eq!(DRM_IOCTL_I915_GET_RESET_STATS, 0xc0186472);
        assert_eq!(size_of::<DrmModeModeinfo>(), 68);
        assert_eq!(size_of::<DrmModeGetConnector>(), 80);
        assert_eq!(size_of::<DrmI915GemExecObject2>(), 56);
    }
}
