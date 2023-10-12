mod ops;
use ops::elementwise::{Add, Div, Mul, Sub};
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
}

fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..(shape.len() - 1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1];
    }
    strides
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
