//! 3-component float vector used by the scene and the host side of the GPU path.

use std::ops::{Add, AddAssign, Index, Mul, Neg, Sub};

#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct V3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

pub const fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3 { x, y, z }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, b: V3) -> V3 {
        v3(self.x + b.x, self.y + b.y, self.z + b.z)
    }
}
impl Sub for V3 {
    type Output = V3;
    fn sub(self, b: V3) -> V3 {
        v3(self.x - b.x, self.y - b.y, self.z - b.z)
    }
}
impl Mul<f32> for V3 {
    type Output = V3;
    fn mul(self, s: f32) -> V3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}
impl Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        self * -1.0
    }
}
impl AddAssign for V3 {
    fn add_assign(&mut self, b: V3) {
        *self = *self + b;
    }
}
impl Index<usize> for V3 {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        match i {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

pub fn dot(a: V3, b: V3) -> f32 {
    a.x * b.x + a.y * b.y + a.z * b.z
}
pub fn cross(a: V3, b: V3) -> V3 {
    v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}
pub fn length(a: V3) -> f32 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    let l = length(a);
    if l > 0.0 { a * (1.0 / l) } else { a }
}
pub fn vmin(a: V3, b: V3) -> V3 {
    v3(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}
pub fn vmax(a: V3, b: V3) -> V3 {
    v3(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}
