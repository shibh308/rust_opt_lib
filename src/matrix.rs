use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;

#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    pub h: usize,
    pub w: usize,
    a: Vec<T>,
}

impl<T> Matrix<T>
where
    T: Default + Copy,
{
    pub fn new(h: usize, w: usize, val: T) -> Self {
        Matrix {
            h,
            w,
            a: vec![val; h * w],
        }
    }
    pub fn transpose(&self) -> Self {
        let mut a = self.a.clone();
        for i in 0..self.h {
            for j in 0..self.w {
                a[j * self.w + i] = self.a[i * self.w + j];
            }
        }
        Matrix {
            h: self.w,
            w: self.h,
            a,
        }
    }
    pub fn concat(&self, rhs: &Self) -> Self {
        assert_eq!(self.h, rhs.h);
        let mut a = vec![Default::default(); self.h * (self.w + rhs.w)];
        for i in 0..self.h {
            for j in 0..self.w {
                a[i * (self.w + rhs.w) + j] = self.get(i, j);
            }
            for j in 0..rhs.w {
                a[i * (self.w + rhs.w) + self.w + j] = rhs.get(i, j);
            }
        }
        Matrix {
            h: self.h,
            w: self.w + rhs.w,
            a,
        }
    }
    pub fn split(&self, idx: usize) -> (Self, Self) {
        assert!(idx < self.w);
        let mut x = Matrix::new(self.h, idx, <T>::default());
        let mut y = Matrix::new(self.h, self.w - idx, <T>::default());
        for i in 0..self.h {
            for j in 0..idx {
                *x.get_mut(i, j) = self.get(i, j);
            }
            for j in 0..(self.w - idx) {
                *y.get_mut(i, j) = self.get(i, j + idx);
            }
        }
        (x, y)
    }
    pub fn get(&self, i: usize, j: usize) -> T {
        assert!(i < self.h);
        assert!(j < self.w);
        self.a[i * self.w + j]
    }
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut T {
        assert!(i < self.h);
        assert!(j < self.w);
        &mut self.a[i * self.w + j]
    }
}

impl<T> Matrix<T>
where
    T: Default + Copy,
    f64: From<T>,
{
    pub fn as_f64(&self) -> Matrix<f64> {
        let a: Vec<f64> = self.a.iter().map(|x| f64::from(*x)).collect();
        Matrix {
            h: self.h,
            w: self.w,
            a,
        }
    }
}

impl Matrix<f64> {
    const EPS: f64 = 1e-6;
    pub fn almost_eq(&self, rhs: &Self) -> bool {
        assert_eq!(self.h, rhs.h);
        assert_eq!(self.w, rhs.w);
        let diff: Vec<_> = self
            .a
            .iter()
            .zip(rhs.a.iter())
            .filter_map(|(x, y)| {
                if x.is_infinite() || y.is_infinite() {
                    None
                } else {
                    Some((x - y).abs())
                }
            })
            .collect();
        diff.len() == self.h * self.w && diff.iter().fold(0.0, |m, v| v.max(m)) <= Self::EPS
    }
    // 掃き出し法で、(rank, 操作後の行列) を返す
    pub fn gaussian_elimination(&self) -> (usize, Matrix<f64>) {
        let mut mat = self.clone();
        let rank = mat._gaussian_elimination();
        (rank, mat)
    }
    fn _gaussian_elimination(&mut self) -> usize {
        let mut rank = 0;
        // 上三角行列を目指す
        let mut hist = Vec::new();
        for i in 0..self.h.min(self.w) {
            let v: Vec<_> = (i..self.h).map(|j| self.get(j, i).abs()).collect();
            let max_val = v.iter().fold(0.0, |m, v| v.max(m));
            let max_idx = v.iter().position(|x| *x == max_val);
            if max_val < Self::EPS {
                continue;
            }
            if rank != max_idx.unwrap() + i {
                let max_idx = max_idx.unwrap() + i;
                // 行の入れ替え
                for j in 0..self.w {
                    let mut tmp = 0.0;
                    std::mem::swap(self.get_mut(rank, j), &mut tmp);
                    std::mem::swap(self.get_mut(max_idx, j), &mut tmp);
                    std::mem::swap(self.get_mut(rank, j), &mut tmp);
                }
            }
            let norm_rate = self.get(rank, i);
            for j in 0..self.w {
                *self.get_mut(rank, j) /= norm_rate;
            }
            // 正規化
            // 加算
            for i2 in 0..self.h {
                if rank == i2 {
                    continue;
                }
                let rate = self.get(i2, i);
                for j in 0..self.w {
                    *self.get_mut(i2, j) = self.get(i2, j) - self.get(rank, j) * rate;
                }
            }
            hist.push((i, rank));
            rank += 1;
        }
        for (target_i, target_j) in hist.iter().rev() {
            let val = self.get(*target_i, *target_j);
            assert!((val - 1.0).abs() < Self::EPS);
            for i in 0..*target_i {
                if i == *target_i {
                    continue;
                }
                let rate = self.get(i, *target_j);
                for j in 0..self.w {
                    *self.get_mut(i, j) = self.get(i, j) - self.get(*target_i, j) * rate;
                }
            }
        }
        rank
    }
    fn inverse_matrix(&self) -> Option<Self> {
        assert_eq!(self.h, self.w);
        let n = self.h;
        let mut right = Matrix::new(n, n, 0.0f64);
        for i in 0..n {
            *right.get_mut(i, i) = 1.0;
        }
        let extend = self.concat(&right);
        let (rank, ex_inv) = extend.gaussian_elimination();
        if rank != n {
            None
        } else {
            let inv = ex_inv.split(n);
            Some(inv.1)
        }
    }
}

impl<T> std::fmt::Debug for Matrix<T>
where
    T: Default + Copy + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut v = vec![vec![Default::default(); self.w]; self.h];
        for i in 0..self.h {
            for j in 0..self.w {
                v[i][j] = self.get(i, j);
            }
        }
        f.debug_struct("Matrix")
            .field("h", &self.h)
            .field("w", &self.w)
            .field("a", &v)
            .finish()
    }
}

impl<T> From<Vec<T>> for Matrix<T>
where
    T: Default + Copy,
{
    fn from(a: Vec<T>) -> Self {
        Matrix {
            h: a.len(),
            w: 1,
            a,
        }
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Default + Copy,
{
    fn from(x: Vec<Vec<T>>) -> Self {
        let h = x.len();
        assert_ne!(h, 0);
        let w = x[0].len();
        for i in 0..h {
            assert_eq!(x[i].len(), w);
        }
        let mut a = vec![Default::default(); h * w];
        for i in 0..h {
            for j in 0..w {
                a[i * w + j] = x[i][j];
            }
        }
        Matrix { h, w, a }
    }
}

#[opimps::impl_ops(Add)]
fn add<T>(self: Matrix<T>, rhs: Matrix<T>) -> Matrix<T>
where
    T: Default + Copy + Add<Output = T>,
{
    assert_eq!(self.h, rhs.h);
    assert_eq!(self.w, rhs.w);
    let mut a = vec![Default::default(); self.h * self.w];
    for i in 0..(self.h * self.w) {
        a[i] = self.a[i] + rhs.a[i];
    }
    Matrix {
        h: self.h,
        w: self.w,
        a: a.clone(),
    }
}

#[opimps::impl_ops(Sub)]
fn sub<T>(self: Matrix<T>, rhs: Self) -> Matrix<T>
where
    T: Default + Copy + Sub<Output = T>,
{
    assert_eq!(self.h, rhs.h);
    assert_eq!(self.w, rhs.w);
    let mut a = vec![Default::default(); self.h * self.w];
    for i in 0..(self.h * self.w) {
        a[i] = self.a[i] - rhs.a[i];
    }
    Matrix {
        h: self.h,
        w: self.w,
        a,
    }
}

#[opimps::impl_ops(Mul)]
fn mul<T>(self: Matrix<T>, rhs: Self) -> Matrix<T>
where
    T: Default + Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(self.w, rhs.h);
    let mut a = vec![Default::default(); self.h * rhs.w];
    for k in 0..self.w {
        for i in 0..self.h {
            for j in 0..rhs.w {
                a[i * rhs.w + j] = a[i * rhs.w + j] + self.a[i * self.w + k] * rhs.a[k * rhs.w + j];
            }
        }
    }
    Matrix {
        h: self.h,
        w: rhs.w,
        a,
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn add_sub() {
        let a = Matrix::from(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]);
        let b = Matrix::new(3, 3, 1);
        let c = Matrix::from(vec![vec![2, 3, 4], vec![5, 6, 7], vec![8, 9, 10]]);
        assert_eq!(&a + &b, c);
        assert_eq!(c - b, a);
    }

    #[test]
    fn mul() {
        let a = Matrix::from(vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
        let b = Matrix::from(vec![vec![2, 1, 0], vec![4, 3, 2]]);
        let c = Matrix::from(vec![vec![10, 22, 34], vec![7, 15, 23], vec![4, 8, 12]]).transpose();
        assert_eq!(a * b, c);
    }

    #[test]
    fn inner_prod() {
        let a = Matrix::from(vec![1, 2, 3]).transpose();
        let b = Matrix::from(vec![4, 5, 6]);
        assert_eq!((a * b).get(0, 0), 32);
    }

    #[test]
    fn row_reduction() {
        let a = [
            vec![
                vec![1., 3., 1., 9.],
                vec![1., 1., -1., 1.],
                vec![3., 11., 5., 35.],
            ],
            vec![
                vec![1., 1., -1., 1., 0., 0.],
                vec![1., -1., 1., 0., 1., 0.],
                vec![-1., 1., 1., 0., 0., 1.],
            ],
            vec![vec![2., 1.], vec![1., 2.], vec![1., 1.]],
            vec![
                vec![2., 4., 2., 2., 8.],
                vec![4., 10., 3., 3., 17.],
                vec![2., 6., 1., 1., 9.],
                vec![3., 7., 1., 4., 11.],
            ],
        ];
        let b = [
            vec![
                vec![1., 0., -2., -3.],
                vec![0., 1., 1., 4.],
                vec![0., 0., 0., 0.],
            ],
            vec![
                vec![1., 0., 0., 0.5, 0.5, 0.0],
                vec![0., 1., 0., 0.5, 0.0, 0.5],
                vec![0., 0., 1., 0.0, 0.5, 0.5],
            ],
            vec![vec![1., 0.], vec![0., 1.], vec![0., 0.]],
            vec![
                vec![1., 0., 0., 4., 1.],
                vec![0., 1., 0., -1., 1.],
                vec![0., 0., 1., -1., 1.],
                vec![0., 0., 0., 0., 0.],
            ],
        ];
        let rank = [2, 3, 2, 3];

        assert_eq!(a.len(), b.len());
        assert_eq!(b.len(), rank.len());
        for ((a, b), rank) in a.iter().zip(b).zip(rank) {
            let a = Matrix::from(a.clone());
            let b = Matrix::from(b.clone());
            let (pred_rank, c) = a.gaussian_elimination();
            assert_eq!(rank, pred_rank);
            assert!(c.almost_eq(&b));
        }
        let a = vec![vec![1, 1], vec![1, 2]];
        let b = vec![vec![2, -1], vec![-1, 1]];
        let inv = Matrix::from(b).as_f64();
        let pred = Matrix::from(a).as_f64().inverse_matrix();
        assert!(pred.is_some());
        assert!(inv.almost_eq(&pred.unwrap()));

        let a = vec![vec![2, 4], vec![-1, -2]];
        let pred = Matrix::from(a).as_f64().inverse_matrix();
        assert!(pred.is_none());
    }
}
