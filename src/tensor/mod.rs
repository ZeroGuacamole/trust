mod ops;
mod utils;

use ops::elementwise::{Add, Div, Mul, Sub};
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

        let new_shape = broadcast_shape(&self.shape, &other.shape);
        let new_strides = compute_strides(&new_shape);
        let mut new_data = Vec::with_capacity(new_shape.iter().product());

        for idx in multi_dim_iter(&new_shape) {
            let value1 = self.get_broadcast_value(&idx);
            let value2 = other.get_broadcast_value(&idx);
            new_data.push(op(value1, value2));
        }

        Ok(Self {
            data: new_data,
            shape: new_shape,
            strides: new_strides,
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

#[cfg(test)]
mod tests {
    use super::*;

    // Tensor manipulation

    #[test]
    fn test_reshape() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let reshaped = tensor.reshape(vec![4, 1]);

        assert!(reshaped.is_ok());
        assert_eq!(reshaped.unwrap().shape, vec![4, 1]);
    }

    #[test]
    fn test_reshape_size_mismatch() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.reshape(vec![3]);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Total number of elements in the new shape must match the original.".to_string()
        );
    }

    #[test]
    fn test_squeeze() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 1, 2]);
        let squeezed = tensor.squeeze();

        assert_eq!(squeezed.shape, vec![2, 2]);
    }

    #[test]
    fn test_squeeze_no_dims_of_one() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let squeezed = tensor.squeeze();

        assert_eq!(squeezed.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(squeezed.shape, vec![2, 2]);
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2]);
        let unsqueezed = tensor.unsqueeze(1);

        assert!(unsqueezed.is_ok());
        assert_eq!(unsqueezed.unwrap().shape, vec![2, 1]);
    }

    #[test]
    fn test_unsqueeze_axis_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2]);
        let result = tensor.unsqueeze(3);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Axis for unsqueeze operation is out of bounds.".to_string()
        );
    }

    // Indexing

    #[test]
    fn test_tensor_indexing() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(tensor[&[0, 1]], 2.0);
        assert_eq!(tensor[&[1, 0]], 3.0);
    }

    #[test]
    fn test_tensor_indexing_few_indices() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = std::panic::catch_unwind(|| tensor[&[1]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_indexing_many_indices() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = std::panic::catch_unwind(|| tensor[&[1, 1, 1]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_indexing_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = std::panic::catch_unwind(|| tensor[&[2, 1]]);
        assert!(result.is_err());
    }

    // Slicing

    #[test]
    fn test_tensor_slicing() {
        let tensor = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        let result = tensor.slice(&[0, 1], &[2, 3]).unwrap();

        assert_eq!(result.data, vec![2.0, 3.0, 5.0, 6.0]);
        assert_eq!(result.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_slicing_few_indices() {
        let tensor: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.slice(&[1], &[2, 2]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Expected 2 start indices, but got 1.");
    }

    #[test]
    fn test_tensor_slicing_many_indices() {
        let tensor: Tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.slice(&[1, 1], &[2, 2, 2]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Expected 2 end indices, but got 3.");
    }

    #[test]
    fn test_tensor_slicing_start_greater_than_end() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.slice(&[1, 2], &[1, 1]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Start index (2) cannot be greater than end index (1)."
        );
    }

    #[test]
    fn test_tensor_slicing_out_of_bounds() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let result = tensor.slice(&[2, 2], &[3, 3]);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Indices out of bounds for dimension of size 2. Got start: 2, end: 3."
        );
    }

    // Element-wise operations

    #[test]
    fn test_tensor_addition() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = tensor1.add(&tensor2);

        assert!(result.is_ok());

        let result_tensor = *result.unwrap();
        assert_eq!(result_tensor.data, vec![5.0, 7.0, 9.0]);
        assert_eq!(result_tensor.shape, vec![3]);
    }

    #[test]
    fn test_tensor_addition_mismatched_shapes() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = tensor1.add(&tensor2);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Shapes of the tensors do not match.".to_string()
        );
    }

    #[test]
    fn test_tensor_subtraction() {
        let tensor1 = Tensor::new(vec![5.0, 5.0, 10.0], vec![3]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = tensor1.sub(&tensor2);

        assert!(result.is_ok());

        let result_tensor = *result.unwrap();
        assert_eq!(result_tensor.data, vec![1.0, 0.0, 4.0]);
        assert_eq!(result_tensor.shape, vec![3]);
    }

    #[test]
    fn test_tensor_subtraction_mismatched_shapes() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = tensor1.sub(&tensor2);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Shapes of the tensors do not match.".to_string()
        );
    }

    #[test]
    fn test_tensor_multiplication() {
        let tensor1 = Tensor::new(vec![5.0, 5.0, 10.0], vec![3]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = tensor1.mul(&tensor2);

        assert!(result.is_ok());

        let result_tensor = *result.unwrap();
        assert_eq!(result_tensor.data, vec![20.0, 25.0, 60.0]);
        assert_eq!(result_tensor.shape, vec![3]);
    }

    #[test]
    fn test_tensor_multiplication_mismatched_shapes() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = tensor1.mul(&tensor2);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Shapes of the tensors do not match.".to_string()
        );
    }

    #[test]
    fn test_tensor_division() {
        let tensor1 = Tensor::new(vec![20.0, 5.0, 12.0], vec![3]);
        let tensor2 = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = tensor1.div(&tensor2);

        assert!(result.is_ok());

        let result_tensor = *result.unwrap();
        assert_eq!(result_tensor.data, vec![5.0, 1.0, 2.0]);
        assert_eq!(result_tensor.shape, vec![3]);
    }

    #[test]
    fn test_tensor_division_mismatched_shapes() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = tensor1.div(&tensor2);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Shapes of the tensors do not match.".to_string()
        );
    }
}
