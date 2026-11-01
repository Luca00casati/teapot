//! Images, PPM output, and DRM/KMS dumb buffers through raw ioctls (no libdrm).

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::os::fd::{AsRawFd, RawFd};

use crate::sys::*;

/// A view of an XRGB8888 image in memory someone else owns: a Vec or a DRM
/// buffer mapping. It is a raw view, like a C pointer + pitch.
#[derive(Clone, Copy)]
pub struct Target {
    pub base: *mut u8,
    pub w: usize,
    pub h: usize,
    pub pitch: usize,
}

impl Target {
    pub fn xrgb(buf: &mut [u32], w: usize, h: usize) -> Target {
        assert!(buf.len() >= w * h);
        Target { base: buf.as_mut_ptr() as *mut u8, w, h, pitch: w * 4 }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn row(&self, y: usize) -> &mut [u32] {
        assert!(y < self.h);
        unsafe { std::slice::from_raw_parts_mut(self.base.add(y * self.pitch) as *mut u32, self.w) }
    }
}

/// copy src to dst, upscaling by an integer factor
pub fn blit(src: &Target, dst: &Target, scale: usize) {
    for y in 0..dst.h {
        let row = src.row((y / scale).min(src.h - 1));
        for (x, d) in dst.row(y).iter_mut().enumerate() {
            *d = row[(x / scale).min(src.w - 1)];
        }
    }
}

pub fn write_ppm(path: &str, t: &Target) -> bool {
    let write = || -> std::io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        write!(f, "P6\n{} {}\n255\n", t.w, t.h)?;
        for y in 0..t.h {
            for &p in t.row(y).iter() {
                f.write_all(&[(p >> 16) as u8, (p >> 8) as u8, p as u8])?;
            }
        }
        f.flush()
    };
    match write() {
        Ok(()) => true,
        Err(e) => {
            eprintln!("{path}: {e}");
            false
        }
    }
}

/* ------------------------------------------------------ DRM / KMS ------ */

/// The first connected output of a DRM card, with two dumb buffers: still
/// images draw into the front one, animation flips between them. Dropping it
/// restores the CRTC and frees the buffers.
pub struct Drm {
    file: File,
    conn: u32,
    crtc: u32,
    fb: [u32; 2],
    handle: [u32; 2],
    size: u64,
    map: [*mut u8; 2],
    front: usize,         // buffer currently scanned out (or queued to be)
    flip_pending: bool,   // a page flip was queued and its completion event not yet read
    saved: DrmModeCrtc,   // the CRTC as we found it
    tgt: Target,
}

impl Drm {
    pub fn open(path: &str) -> Option<Drm> {
        let file = OpenOptions::new().read(true).write(true).open(path).ok()?;
        let fd = file.as_raw_fd();
        ioctl_none(fd, DRM_IOCTL_SET_MASTER); // implicit on first open from a VT; harmless otherwise

        let mut res = DrmModeCardRes::default();
        let (mut conns, mut crtcs, mut encs) = ([0u32; 64], [0u32; 32], [0u32; 64]);
        if !ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) {
            return None;
        }
        res.count_connectors = res.count_connectors.min(64);
        res.count_crtcs = res.count_crtcs.min(32);
        res.count_encoders = res.count_encoders.min(64);
        res.count_fbs = 0;
        res.connector_id_ptr = conns.as_mut_ptr() as u64;
        res.crtc_id_ptr = crtcs.as_mut_ptr() as u64;
        res.encoder_id_ptr = encs.as_mut_ptr() as u64;
        if !ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) {
            return None;
        }
        // dropping `d` cleans up from here on
        let mut d = Drm {
            file,
            conn: 0,
            crtc: 0,
            fb: [0; 2],
            handle: [0; 2],
            size: 0,
            map: [std::ptr::null_mut(); 2],
            front: 0,
            flip_pending: false,
            saved: DrmModeCrtc::default(),
            tgt: Target { base: std::ptr::null_mut(), w: 0, h: 0, pitch: 0 },
        };

        for &conn in &conns[..res.count_connectors as usize] {
            let mut c = DrmModeGetConnector { connector_id: conn, ..Default::default() };
            if !ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut c) {
                continue; // (probes modes)
            }
            if c.connection != 1 || c.count_modes == 0 {
                continue;
            }
            let mut modes = vec![DrmModeModeinfo::default(); c.count_modes as usize];
            let mut cenc = vec![0u32; c.count_encoders as usize + 1];
            c.modes_ptr = modes.as_mut_ptr() as u64;
            c.encoders_ptr = cenc.as_mut_ptr() as u64;
            c.count_props = 0;
            c.props_ptr = 0;
            c.prop_values_ptr = 0;
            if !ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &mut c) || c.count_modes == 0 {
                continue;
            }
            modes.truncate(c.count_modes as usize);
            let mode = *modes.iter().find(|m| m.type_ & DRM_MODE_TYPE_PREFERRED != 0).unwrap_or(&modes[0]);

            let mut crtc = 0;
            if c.encoder_id != 0 {
                let mut e = DrmModeGetEncoder { encoder_id: c.encoder_id, ..Default::default() };
                if ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &mut e) {
                    crtc = e.crtc_id;
                }
            }
            for &enc in cenc.iter().take(c.count_encoders as usize) {
                if crtc != 0 {
                    break;
                }
                let mut e = DrmModeGetEncoder { encoder_id: enc, ..Default::default() };
                if !ioctl(fd, DRM_IOCTL_MODE_GETENCODER, &mut e) {
                    continue;
                }
                if let Some(j) = (0..res.count_crtcs as usize).find(|j| e.possible_crtcs & (1 << j) != 0) {
                    crtc = crtcs[j];
                }
            }
            if crtc == 0 {
                continue;
            }

            let mut cd = DrmModeCreateDumb::default();
            for k in 0..2 {
                cd = DrmModeCreateDumb { width: mode.hdisplay as u32, height: mode.vdisplay as u32, bpp: 32, ..Default::default() };
                if !ioctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &mut cd) {
                    perror("CREATE_DUMB");
                    return None;
                }
                d.handle[k] = cd.handle;
                d.size = cd.size;
                let mut fb = DrmModeFbCmd { width: cd.width, height: cd.height, pitch: cd.pitch, bpp: 32, depth: 24, handle: cd.handle, ..Default::default() };
                if !ioctl(fd, DRM_IOCTL_MODE_ADDFB, &mut fb) {
                    perror("ADDFB");
                    return None;
                }
                d.fb[k] = fb.fb_id;
                let mut md = DrmModeMapDumb { handle: cd.handle, ..Default::default() };
                if !ioctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &mut md) {
                    perror("MAP_DUMB");
                    return None;
                }
                let Some(map) = mmap(fd, cd.size, md.offset) else {
                    perror("mmap dumb");
                    return None;
                };
                unsafe { std::ptr::write_bytes(map, 0, cd.size as usize) };
                d.map[k] = map;
            }

            d.saved = DrmModeCrtc { crtc_id: crtc, ..Default::default() };
            ioctl(fd, DRM_IOCTL_MODE_GETCRTC, &mut d.saved);

            d.conn = conn;
            let mut set = DrmModeCrtc {
                set_connectors_ptr: &d.conn as *const u32 as u64,
                count_connectors: 1,
                crtc_id: crtc,
                fb_id: d.fb[0],
                mode_valid: 1,
                mode,
                ..Default::default()
            };
            if !ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &mut set) {
                perror("SETCRTC (is a compositor holding DRM master? run from a text VT)");
                d.saved.mode_valid = 0;
                return None;
            }
            d.crtc = crtc;
            d.tgt = Target { base: std::ptr::null_mut(), w: cd.width as usize, h: cd.height as usize, pitch: cd.pitch as usize };
            eprintln!("drm: {path} connector {conn} crtc {crtc} {}x{}@{}", mode.hdisplay, mode.vdisplay, mode.vrefresh);
            return Some(d);
        }
        eprintln!("drm: {path} has no usable connected output");
        None
    }

    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
    fn target(&self, k: usize) -> Target {
        Target { base: self.map[k], ..self.tgt }
    }
    pub fn front(&self) -> Target {
        self.target(self.front)
    }
    pub fn back(&self) -> Target {
        self.target(1 - self.front)
    }

    fn dirty_buf(&self, k: usize) {
        let mut d = DrmModeFbDirtyCmd { fb_id: self.fb[k], ..Default::default() };
        ioctl(self.fd(), DRM_IOCTL_MODE_DIRTYFB, &mut d);
    }
    /// the front buffer changed (needed by virtio/udl style drivers)
    pub fn dirty(&self) {
        self.dirty_buf(self.front);
    }

    /// until the last queued flip happened, so the back buffer is free
    pub fn wait_flip(&mut self) {
        if !self.flip_pending {
            return;
        }
        let mut ev = [0u8; 1024];
        if poll_in(self.fd(), 100) {
            let _ = (&self.file).read(&mut ev); // event content unused
        }
        self.flip_pending = false;
    }

    /// queue the back buffer for the next vblank (does not wait)
    pub fn flip(&mut self) {
        let back = 1 - self.front;
        self.dirty_buf(back);
        let mut fl = DrmModeCrtcPageFlip { crtc_id: self.crtc, fb_id: self.fb[back], flags: DRM_MODE_PAGE_FLIP_EVENT, ..Default::default() };
        if ioctl(self.fd(), DRM_IOCTL_MODE_PAGE_FLIP, &mut fl) {
            self.flip_pending = true;
        } else {
            // no page flip support: just switch the CRTC over
            let mut c = DrmModeCrtc { crtc_id: self.crtc, ..Default::default() };
            ioctl(self.fd(), DRM_IOCTL_MODE_GETCRTC, &mut c);
            c.fb_id = self.fb[back];
            c.set_connectors_ptr = &self.conn as *const u32 as u64;
            c.count_connectors = 1;
            ioctl(self.fd(), DRM_IOCTL_MODE_SETCRTC, &mut c);
        }
        self.front = back;
    }
}

impl Drop for Drm {
    fn drop(&mut self) {
        self.wait_flip();
        let fd = self.fd();
        if self.saved.mode_valid != 0 {
            self.saved.set_connectors_ptr = &self.conn as *const u32 as u64;
            self.saved.count_connectors = 1;
            ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &mut self.saved);
        }
        for k in 0..2 {
            if !self.map[k].is_null() {
                munmap(self.map[k], self.size);
            }
            if self.fb[k] != 0 {
                ioctl(fd, DRM_IOCTL_MODE_RMFB, &mut self.fb[k]);
            }
            if self.handle[k] != 0 {
                let mut dd = DrmModeDestroyDumb { handle: self.handle[k] };
                ioctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &mut dd);
            }
        }
        // the file closes itself
    }
}
