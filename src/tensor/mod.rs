pub mod ops;
mod utils;

use ops::elementwise::{Add, Div, Mul, Sub};
use ops::reductions::Sum;
use std::ops::Index;
use utils::{
    are_broadcast_compatible, broadcast_shape, compute_strides, multi_dim_iter, recursive_slice,
};

/*
    Notes:
    - Storing the data as 1D for memory management simplification
    and efficient utilization of cache
    - Shape will determine the dimensions of the tensor
    - Strides for conmputing the position of an element in flat array
    based on its multi-dimensional index
*/
#[derive(Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Creates a new Tensor with the given data and shape.
    ///
    /// # Arguments
    ///
    /// * `data` - A flat vector containing the tensor's data.
    /// * `shape` - A vector describing the size of each tensor dimension.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Ensures that the shapes of two tensors match.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to compare with the current one.
    ///
    /// # Returns
    ///
    /// * `Result<(), String>` - Ok if shapes match, otherwise an error string describing the mismatch.
    fn ensure_same_shape(&self, other: &Self) -> Result<(), String> {
        if self.shape != other.shape {
            Err("Shapes of the tensors do not match.".to_string())
        } else {
            Ok(())
        }
    }

    /// Reshapes the tensor to the specified shape.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The desired shape for the reshaped tensor.
    ///
    /// # Returns
    ///
    /// * `Result<Tensor, String>` - The reshaped tensor if the total number of elements remains constant, otherwise an error string.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let current_total_size: usize = self.shape.iter().product();
        let new_total_size: usize = new_shape.iter().product();

        if new_total_size != current_total_size {
            return Err(
                "Total number of elements in the new shape must match the original.".to_string(),
            );
        }

        let new_strides = compute_strides(&new_shape);

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Removes dimensions of size one from the tensor's shape.
    ///
    /// This operation returns a tensor with the same data but with dimensions of size one removed.
    /// For example, a tensor with shape `[1, 3, 1, 2]` will be squeezed to `[3, 2]`.
    ///
    /// # Returns
    ///
    /// * `Tensor` - The squeezed tensor.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().cloned().filter(|&dim| dim > 1).collect();
        let new_strides = compute_strides(&new_shape);
        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Adds a dimension of size one at the specified position in the tensor's shape.
    ///
    /// # Arguments
    ///
    /// * `dim` - The position at which to add the new dimension.
    ///
    /// # Returns
    ///
    /// * `Result<Tensor, String>` - The tensor with the expanded dimension if `dim` is valid, otherwise an error string.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, String> {
        if axis > self.shape.len() {
            return Err("Axis for unsqueeze operation is out of bounds.".to_string());
        }

        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        let new_strides = compute_strides(&new_shape);

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Extracts a sub-tensor based on specified start and end points for each dimension.
    ///
    /// # Arguments
    ///
    /// * `starts` - The starting indices for each dimension.
    /// * `ends` - The ending indices for each dimension.
    ///
    /// # Returns
    ///
    /// * `Result<Tensor, String>` - The sliced tensor if indices are valid, otherwise an error string.
    pub fn slice(&self, starts: &[usize], ends: &[usize]) -> Result<Self, String> {
        if starts.len() != self.shape.len() {
            return Err(format!(
                "Expected {} start indices, but got {}.",
                self.shape.len(),
                starts.len()
            ));
        }
        if ends.len() != self.shape.len() {
            return Err(format!(
                "Expected {} end indices, but got {}.",
                self.shape.len(),
                ends.len()
            ));
        }

        for ((&start, &end), &dim) in starts.iter().zip(ends.iter()).zip(self.shape.iter()) {
            if start > end {
                return Err(format!(
                    "Start index ({}) cannot be greater than end index ({}).",
                    start, end
                ));
            }
            if start >= dim || end > dim {
                return Err(format!(
                    "Indices out of bounds for dimension of size {}. Got start: {}, end: {}.",
                    dim, start, end
                ));
            }
        }

        let new_shape: Vec<usize> = starts
            .iter()
            .zip(ends.iter())
            .map(|(&start, &end)| end - start)
            .collect();

        // We might need to change to a more efficient algorithms or strategies,
        // for cases like, very high-dimensional tensors or large slices
        let new_data = recursive_slice(
            &self.data,
            &self.shape,
            &self.strides,
            starts,
            ends,
            self.shape.len(),
        );
        let new_strides = compute_strides(&new_shape);

        Ok(Self {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
        })
    }

    // Broadcasting

    fn apply_broadcast<F>(&self, other: &Self, op: F) -> Result<Self, String>
    where
        F: Fn(f32, f32) -> f32,
    {
        if !are_broadcast_compatible(&self.shape, &other.shape) {
            return Err("Shapes of the tensors are not broadcast compatible.".to_string());
        }

        let result_shape = broadcast_shape(&self.shape, &other.shape);
        let reslt_strides = compute_strides(&result_shape);
        let mut result_data = Vec::with_capacity(result_shape.iter().product());

        for idx in multi_dim_iter(&result_shape) {
            let value1 = self.get_broadcast_value(&idx);
            let value2 = other.get_broadcast_value(&idx);
            result_data.push(op(value1, value2));
        }

        Ok(Self {
            data: result_data,
            shape: result_shape,
            strides: reslt_strides,
        })
    }

    fn get_broadcast_value(&self, idx: &[usize]) -> f32 {
        let mut true_idx = Vec::with_capacity(self.shape.len());

        for (i, &dim) in self.shape.iter().enumerate() {
            let offset = idx.len() - self.shape.len();
            true_idx.push(if dim == 1 { 0 } else { idx[i + offset] });
        }

        self[&true_idx]
    }

    pub fn add_broadcast(&self, other: &Self) -> Result<Self, String> {
        self.apply_broadcast(other, |a, b| a + b)
    }

    pub fn sub_broadcast(&self, other: &Self) -> Result<Self, String> {
        self.apply_broadcast(other, |a, b| a - b)
    }

    pub fn mul_broadcast(&self, other: &Self) -> Result<Self, String> {
        self.apply_broadcast(other, |a, b| a * b)
    }

    pub fn div_broadcast(&self, other: &Self) -> Result<Self, String> {
        if other.data.iter().any(|&value| value == 0.0) {
            return Err("Division by zero.".to_string());
        }
        self.apply_broadcast(other, |a, b| a / b)
    }
}

impl Index<&[usize]> for Tensor {
    type Output = f32;

    fn index(&self, idx: &[usize]) -> &Self::Output {
        assert_eq!(
            idx.len(),
            self.shape.len(),
            "Incorrect number of indices provided."
        );

        let mut flat_idx = 0;
        for (i, &index) in idx.iter().enumerate() {
            flat_idx += index * self.strides[i];
        }

        &self.data[flat_idx]
    }
}

impl Add for Tensor {
    /// Computes the element-wise addition of two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to add.
    ///
    /// # Returns
    ///
    /// * `Result<Box<Tensor>, String>` - The resulting tensor if shapes match, otherwise an error string.
    fn add(&self, other: &Self) -> Result<Box<Self>, String> {
        self.ensure_same_shape(other)?;

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();

        Ok(Box::new(Self::new(new_data, self.shape.clone())))
    }
}

impl Sub for Tensor {
    /// Computes the element-wise subtraction of two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to subtract from the current one.
    ///
    /// # Returns
    ///
    /// * `Result<Box<Tensor>, String>` - The resulting tensor if shapes match, otherwise an error string.
    fn sub(&self, other: &Self) -> Result<Box<Self>, String> {
        self.ensure_same_shape(other)?;

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        Ok(Box::new(Self::new(new_data, self.shape.clone())))
    }
}

impl Mul for Tensor {
    /// Computes the element-wise multiplication of two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to multiply with the current one.
    ///
    /// # Returns
    ///
    /// * `Result<Box<Tensor>, String>` - The resulting tensor if shapes match, otherwise an error string.
    fn mul(&self, other: &Self) -> Result<Box<Self>, String> {
        self.ensure_same_shape(other)?;

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();

        Ok(Box::new(Self::new(new_data, self.shape.clone())))
    }
}

impl Div for Tensor {
    /// Computes the element-wise division of two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to divide the current one by.
    ///
    /// # Returns
    ///
    /// * `Result<Box<Tensor>, String>` - The resulting tensor if shapes match and no division by zero occurs, otherwise an error string.
    fn div(&self, other: &Self) -> Result<Box<Self>, String> {
        self.ensure_same_shape(other)?;

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a / b)
            .collect();

        Ok(Box::new(Self::new(new_data, self.shape.clone())))
    }
}

impl Sum for Tensor {
    fn sum(&self, dim: Option<usize>) -> Self {
        if dim.is_none() {
            let total_sum = self.data.iter().sum();
            return Self::new(vec![total_sum], vec![1]);
        }

        let dim = dim.unwrap();
        let mut result_shape = self.shape.clone();
        result_shape[dim] = 1;

        let mut result_data = vec![0.0; result_shape.iter().product()];
        let result_strides = compute_strides(&result_shape);

        // TODO: Might need to revisit as it will introduced overhead
        // overhead because we're repeatedly converting between
        // flat and multi-dimensional indices
        for i in 0..self.shape[dim] {
            for (j, result_val) in result_data.iter_mut().enumerate() {
                let mut idx = flat_to_multi_index(&self.shape, &self.strides, j);
                idx[dim] = i;

                let flat_index = compute_flat_index(&self.shape, &self.strides, &idx);
                *result_val += self.data[flat_index];
            }
        }

        Self {
            data: result_data,
            shape: result_shape,
            strides: result_strides,
        }
    }
}

fn compute_flat_index(shape: &[usize], strides: &[usize], idx: &[usize]) -> usize {
    assert_eq!(shape.len(), idx.len());
    idx.iter().zip(strides).fold(0, |acc, (&i, &s)| acc + i * s)
}

fn flat_to_multi_index(shape: &[usize], strides: &[usize], flat_index: usize) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    let mut remainder = flat_index;

    for i in (0..shape.len()).rev() {
        idx[i] = remainder / strides[i];
        remainder %= strides[i];
    }

    idx
}
