#![allow(dead_code, unused)]
#![forbid(unused_must_use)]

use shrinkwraprs::Shrinkwrap;
use std::ops::*;

#[derive(Debug, Shrinkwrap)]
#[shrinkwrap(mutable)]
struct Matrix<Scalar, const ROWS: usize, const COLUMNS: usize>([[Scalar; COLUMNS]; ROWS]);

impl<Scalar, const ROWS: usize, const COLUMNS: usize> Matrix<Scalar, ROWS, COLUMNS> {
    fn size(&self) -> (usize, usize) {
        (ROWS, COLUMNS)
    }
}

type RowVector<Scalar, const COLUMNS: usize> = Matrix<Scalar, 1, COLUMNS>;
type ColumnVector<Scalar, const ROWS: usize> = Matrix<Scalar, ROWS, 1>;
type SquareMatrix<Scalar, const DIMENSION: usize> = Matrix<Scalar, DIMENSION, DIMENSION>;
type Vector<Scalar, const DIMENSION: usize> = ColumnVector<Scalar, DIMENSION>;

impl<Scalar, const ROWS: usize, const COLUMNS: usize> Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Clone,
{
    fn combine_component_wise(self, rhs: Self, f: impl Fn(Scalar, Scalar) -> Scalar) -> Self {
        Self::generate(|row_index, column_index| {
            f(
                self[row_index][column_index].clone(),
                rhs[row_index][column_index].clone(),
            )
        })
    }

    fn transpose(&self) -> Self {
        Self::generate(|row_index, column_index| self[column_index][row_index].clone())
    }

    fn column(&self, index: usize) -> [Scalar; ROWS] {
        std::array::from_fn(|row_index| self[row_index][index].clone())
    }

    fn row(&self, index: usize) -> [Scalar; COLUMNS] {
        self[index].clone()
    }
}

impl<Scalar, const ROWS: usize, const COLUMNS: usize> Matrix<Scalar, ROWS, COLUMNS> {
    fn generate(f: impl Fn(usize, usize) -> Scalar) -> Self {
        Self(std::array::from_fn(|row_index| {
            std::array::from_fn(|column_index| f(row_index, column_index))
        }))
    }
}

fn dot<Scalar: Zero>(lhs: impl Iterator<Item = Scalar>, rhs: impl Iterator<Item = Scalar>) -> Scalar
where
    Scalar: Add<Output = Scalar> + Mul<Output = Scalar> + Clone,
{
    lhs.zip(rhs)
        .map(|(x, y)| x * y)
        .fold(Scalar::zero(), Scalar::add)
}

// equality
impl<Scalar, const ROWS: usize, const COLUMNS: usize> PartialEq for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter()
            .zip(other.iter())
            .all(|(row, rhs_row)| row.iter().zip(rhs_row.iter()).all(|(x, y)| x == y))
    }
}

impl<Scalar, const ROWS: usize, const COLUMNS: usize> Neg for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Neg<Output = Scalar> + Clone,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::generate(|row_index, column_index| -self[row_index][column_index].clone())
    }
}

impl<Scalar, const DIMENSION: usize> SquareMatrix<Scalar, DIMENSION>
where
    Scalar: Clone + PartialEq + Neg<Output = Scalar>,
{
    // check if symmetric
    fn is_symmetric(&self) -> bool {
        let transposed = self.transpose();
        self == &transposed
    }

    // check if skew-symmetric
    fn is_skew_symmetric(&self) -> bool {
        let negative_transposed = -self.transpose();
        self == &negative_transposed
    }
}

// Add
impl<Scalar, const ROWS: usize, const COLUMNS: usize> Add for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Add<Output = Scalar> + Clone,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.combine_component_wise(rhs, Scalar::add)
    }
}

use num_traits::Zero;

// Additive identity
impl<Scalar, const ROWS: usize, const COLUMNS: usize> Zero for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Zero + Clone,
{
    fn zero() -> Self {
        Self(std::array::from_fn(|_| {
            std::array::from_fn(|_| Scalar::zero())
        }))
    }

    fn is_zero(&self) -> bool {
        self.iter().all(|row| row.iter().all(|x| x.is_zero()))
    }
}

// Scalar multiplication
impl<Scalar, const ROWS: usize, const COLUMNS: usize> Mul<Scalar> for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Mul<Output = Scalar> + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Self::generate(|row_index, column_index| {
            self[row_index][column_index].clone() * rhs.clone()
        })
    }
}

// Matrix substraction
impl<Scalar, const ROWS: usize, const COLUMNS: usize> Sub for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Sub<Output = Scalar> + Clone,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.combine_component_wise(rhs, Scalar::sub)
    }
}

// Matrix multiplication
impl<Scalar, const ROWS: usize, const COLUMNS: usize, const RHS_COLUMNS: usize>
    Mul<Matrix<Scalar, COLUMNS, RHS_COLUMNS>> for Matrix<Scalar, ROWS, COLUMNS>
where
    Scalar: Zero + Add<Output = Scalar> + Mul<Output = Scalar> + Clone,
{
    type Output = Matrix<Scalar, ROWS, RHS_COLUMNS>;

    fn mul(self, rhs: Matrix<Scalar, COLUMNS, RHS_COLUMNS>) -> Self::Output {
        Self::Output::generate(|row_index, column_index| {
            let row = self.row(row_index);
            let column = rhs.column(column_index);
            dot(row.into_iter(), column.into_iter())
        })
    }
}

// multiplicative identity
use num_traits::One;

impl<Scalar, const DIMENSION: usize> One for SquareMatrix<Scalar, DIMENSION>
where
    Scalar: Zero + One + Add<Output = Scalar> + Mul<Output = Scalar> + Clone,
{
    fn one() -> Self {
        Self::generate(|row_index, column_index| {
            if row_index == column_index {
                Scalar::one()
            } else {
                Scalar::zero()
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut m = Matrix([[1, 2], [3, 4]]);
        m[0][0] = 10;
        println!("{:?}", m.iter());
    }
}
