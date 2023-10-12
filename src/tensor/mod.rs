mod ops;
mod utils;

use ops::elementwise::{Add, Div, Mul, Sub};
use std::ops::Index;
use utils::{compute_strides, recursive_slice};

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
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
        }
    }

    fn ensure_same_shape(&self, other: &Self) -> Result<(), String> {
        if self.shape != other.shape {
            Err("Shapes of the tensors do not match.".to_string())
        } else {
            Ok(())
        }
    }

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

    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().cloned().filter(|&dim| dim > 1).collect();
        let new_strides = compute_strides(&new_shape);
        Self {
            data: self.data.clone(),
            shape: new_shape,
            strides: new_strides,
        }
    }

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
