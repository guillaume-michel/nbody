use num::{Float, Zero};
use rand::{distributions::Uniform, Rng};
use rayon::prelude::*;
use std::iter;

use cudart_sys::{error::*, memory::*, stream::*};

#[derive(Debug, Copy, Clone)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Float> Vec3<T> {
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self {
            x: T::zero(),
            y: T::zero(),
            z: T::zero(),
        }
    }

    pub fn norm(&self) -> T {
        self.norm2().sqrt()
    }

    pub fn norm2(&self) -> T {
        self.x * self.x + self.y * self.y + self.z * self.z
    }
}

impl<T: Float> std::ops::Add for Vec3<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Float> std::ops::Sub for Vec3<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Float> std::ops::Mul<T> for Vec3<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

pub fn distance2<T: Float>(p1: Vec3<T>, p2: Vec3<T>) -> T {
    (p1 - p2).norm2()
}

pub fn distance<T: Float>(p1: Vec3<T>, p2: Vec3<T>) -> T {
    (p1 - p2).norm()
}

pub fn generate_uniform_random<T, R>(n: usize, min_value: T, max_value: T, rng: &mut R) -> Vec<T>
where
    R: Rng,
    T: rand::distributions::uniform::SampleUniform,
{
    let range = Uniform::new(min_value, max_value);
    (0..n).map(|_| rng.sample(&range)).collect()
}

pub fn generate_uniform_random_vec3<T, R>(
    n: usize,
    min_value: T,
    max_value: T,
    rng: &mut R,
) -> Vec<Vec3<T>>
where
    R: Rng,
    T: rand::distributions::uniform::SampleUniform,
{
    let range = Uniform::new(min_value, max_value);
    (0..n)
        .map(|_| Vec3 {
            x: rng.sample(&range),
            y: rng.sample(&range),
            z: rng.sample(&range),
        })
        .collect()
}

pub fn compute_gravity<T>(p1: Vec3<T>, m1: T, p2: Vec3<T>, m2: T, softening: T, g: T) -> Vec3<T>
where
    T: Float,
{
    let r = distance(p1, p2);

    if r == T::zero() {
        return Vec3::zero();
    }

    let u = (p2 - p1) * (T::one() / r);
    u * (m1 * m2 * g / (r * r + softening))
}

pub fn compute_accelerations<T>(
    positions: &[Vec3<T>],
    masses: &[T],
    accelerations: &mut [Vec3<T>],
    softening: T,
    g: T,
) where
    T: Float + Send + Sync,
{
    positions
        .par_iter()
        .zip(masses.par_iter())
        .zip(accelerations.par_iter_mut())
        .for_each(|((pi, mi), ai)| {
            let mut acceleration = Vec3::<T>::zero();

            positions.iter().zip(&masses[..]).for_each(|(pj, mj)| {
                acceleration = acceleration + compute_gravity(*pi, *mi, *pj, *mj, softening, g);
            });

            *ai = acceleration;
        });
}

pub fn kinematic<T>(position: Vec3<T>, velocity: Vec3<T>, acceleration: Vec3<T>, dt: T) -> Vec3<T>
where
    T: Float,
{
    position + velocity * dt + acceleration * (dt * dt) // missing 1/2?
}

pub fn move_particules<T>(
    positions: &mut [Vec3<T>],
    velocities: &mut [Vec3<T>],
    accelerations: &[Vec3<T>],
    dt: T,
) where
    T: Float + Send + Sync,
{
    positions
        .par_iter_mut()
        .zip(accelerations.par_iter())
        .zip(velocities.par_iter_mut())
        .for_each(|((p, a), v)| {
            let ai = *a;
            let vi = ai * dt + *v;
            *p = kinematic(*p, vi, ai, dt);
            *v = vi;
        });
}

fn run_cpu() {
    let mut rng = rand::thread_rng();

    let num_steps = 1000usize;

    let g = 1f64;
    let softening = 1f64;

    let num_bodies = 1000usize;

    let min_value = -2f64;
    let max_value = 2f64;

    let dt = 0.01f64;

    let mut positions = generate_uniform_random_vec3(num_bodies, min_value, max_value, &mut rng);
    let mut velocities = generate_uniform_random_vec3(num_bodies, min_value, max_value, &mut rng);
    let mut accelerations =
        generate_uniform_random_vec3(num_bodies, min_value, max_value, &mut rng);

    let masses: Vec<f64> = iter::repeat(1f64).take(num_bodies).collect();

    // println!("{:?}", positions);
    // println!("{:?}", velocities);
    // println!("{:?}", accelerations);

    for _ in 0..num_steps {
        compute_accelerations(&positions, &masses, &mut accelerations, softening, g);

        move_particules(&mut positions, &mut velocities, &accelerations, dt);
    }

    // println!("-----------------------------------------------------");
    // println!("{:?}", positions);
    // println!("{:?}", velocities);
    // println!("{:?}", accelerations);
}

fn run_gpu() -> CudaResult<()> {
    let mut rng = rand::thread_rng();

    let num_steps = 2usize;

    let _g = 1f64;
    let softening = 1f64;

    let num_bodies = 1024000usize;

    let min_value = -2f64;
    let max_value = 2f64;

    let dt = 0.01f64;

    let block_size = 384usize;

    let stream = CudaStream::create_with_flags(cudart_sys::ffi::cudaStreamNonBlocking)?;

    let mut h_positions = generate_uniform_random(4 * num_bodies, min_value, max_value, &mut rng);
    let h_velocities = generate_uniform_random(4 * num_bodies, min_value, max_value, &mut rng);

    unsafe {
        let mut d_positions1: DeviceBuffer<f64> = cuda_malloc_async(num_bodies * 4, &stream)?;
        let mut d_positions2: DeviceBuffer<f64> = cuda_malloc_async(num_bodies * 4, &stream)?;
        let mut d_velocities: DeviceBuffer<f64> = cuda_malloc_async(num_bodies * 4, &stream)?;

        cuda_memcpy_h2d_async(&mut d_positions1, &h_positions, &stream)?;
        cuda_memcpy_h2d_async(&mut d_velocities, &h_velocities, &stream)?;

        nbody_cuda_kernels::cuda::set_softening_squared_f64(softening, &stream)?;

        for _ in 0..num_steps / 2 {
            nbody_cuda_kernels::cuda::integrate_nbody_system_3d_f64(
                &d_positions1,
                &mut d_velocities,
                num_bodies,
                &mut d_positions2,
                dt,
                1f64,
                block_size,
                &stream,
            )?;

            nbody_cuda_kernels::cuda::integrate_nbody_system_3d_f64(
                &d_positions2,
                &mut d_velocities,
                num_bodies,
                &mut d_positions1,
                dt,
                1f64,
                block_size,
                &stream,
            )?;
        }
        cuda_memcpy_d2h_async(&mut h_positions, &d_positions2, &stream)?;

        cuda_free_async(d_positions1, &stream)?;
        cuda_free_async(d_positions2, &stream)?;
        cuda_free_async(d_velocities, &stream)?;

        stream.synchronize()?;
    }

    // println!("{:?}", h_positions);

    Ok(())
}

fn main() {
    run_gpu().unwrap();

    run_cpu();
}
