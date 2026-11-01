//! Passes and frames are averaged in linear light: 8-bit display values in
//! through `lin8`, 12-bit linear back out through `enc12`.

use std::sync::OnceLock;

pub struct ToneLuts {
    pub lin8: [f32; 256],
    pub enc12: [u8; 4096],
}

impl ToneLuts {
    fn new() -> ToneLuts {
        let mut t = ToneLuts { lin8: [0.0; 256], enc12: [0; 4096] };
        for (i, v) in t.lin8.iter_mut().enumerate() {
            *v = (i as f32 / 255.0).powf(2.2);
        }
        for (i, v) in t.enc12.iter_mut().enumerate() {
            *v = ((i as f32 / 4095.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        }
        t
    }

    /// XRGB8888 from linear c[0..3] in [0, 1], scaled by k / 4095
    pub fn encode(&self, c: &[f32], k: f32) -> u32 {
        let e = |v: f32| self.enc12[(v * k) as usize] as u32;
        e(c[0]) << 16 | e(c[1]) << 8 | e(c[2])
    }
}

pub fn tone() -> &'static ToneLuts {
    static T: OnceLock<ToneLuts> = OnceLock::new();
    T.get_or_init(ToneLuts::new)
}
