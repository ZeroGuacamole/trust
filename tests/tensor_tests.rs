#[cfg(test)]
mod tests {
    use trust::tensor::ops::elementwise::*;
    use trust::tensor::*;

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

    // Broadcasting

    #[test]
    fn test_broadcast_addition() {
        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let tensor2 = Tensor::new(vec![1.0], vec![1]);
        let result = tensor1.add_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0]);
        assert_eq!(result.shape, vec![3]);

        let tensor1 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]);
        let tensor2 = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let result = tensor1.add_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
        assert_eq!(result.shape, vec![3, 2]);
    }

    #[test]
    fn test_broadcast_subtraction() {
        let tensor1 = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let tensor2 = Tensor::new(vec![1.0], vec![1]);
        let result = tensor1.sub_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.shape, vec![3]);

        let tensor1 = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]);
        let tensor2 = Tensor::new(vec![1.0, 2.0], vec![1, 2]);
        let result = tensor1.sub_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0]);
        assert_eq!(result.shape, vec![3, 2]);
    }

    #[test]
    fn test_broadcast_multiplication() {
        let tensor1 = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let tensor2 = Tensor::new(vec![2.0], vec![1]);
        let result = tensor1.mul_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![4.0, 6.0, 8.0]);
        assert_eq!(result.shape, vec![3]);

        let tensor1 = Tensor::new(vec![2.0, 4.0, 6.0], vec![3, 1]);
        let tensor2 = Tensor::new(vec![2.0, 3.0], vec![1, 2]);
        let result = tensor1.mul_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![4.0, 6.0, 8.0, 12.0, 12.0, 18.0]);
        assert_eq!(result.shape, vec![3, 2]);
    }

    #[test]
    fn test_broadcast_division() {
        let tensor1: Tensor = Tensor::new(vec![4.0, 6.0, 8.0], vec![3]);
        let tensor2 = Tensor::new(vec![2.0], vec![1]);
        let result = tensor1.div_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![2.0, 3.0, 4.0]);
        assert_eq!(result.shape, vec![3]);

        let tensor1 = Tensor::new(vec![4.0, 8.0, 12.0], vec![3, 1]);
        let tensor2 = Tensor::new(vec![2.0, 4.0], vec![1, 2]);
        let result = tensor1.div_broadcast(&tensor2).unwrap();
        assert_eq!(result.data, vec![2.0, 1.0, 4.0, 2.0, 6.0, 3.0]);
        assert_eq!(result.shape, vec![3, 2]);
    }

    #[test]
    fn test_broadcast_division_by_zero() {
        let tensor1 = Tensor::new(vec![4.0, 6.0, 8.0], vec![3]);
        let tensor2 = Tensor::new(vec![0.0], vec![1]);
        let result = tensor1.div_broadcast(&tensor2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Division by zero.");
    }

    #[test]
    fn test_broadcast_capability_error() {
        let tensor1 = Tensor::new(vec![1.0, 2.0], vec![2]);
        let tensor2 = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let result = tensor1.add_broadcast(&tensor2);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Shapes of the tensors are not broadcast compatible."
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
