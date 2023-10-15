use std::cmp::max;

/// Computes the strides for a given tensor shape.
///
/// # Arguments
///
/// * `shape` - The shape of the tensor.
///
/// # Returns
///
/// * `Vec<usize>` - The computed strides.
pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..(shape.len() - 1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1];
    }
    strides
}

/// Recursively computes the data for a sliced tensor.
///
/// # Arguments
///
/// * `tensor_data` - The flat data of the tensor.
/// * `tensor_shape` - The shape of the tensor.
/// * `tensor_strides` - The strides of the tensor.
/// * `starts` - The starting indices for each dimension of the slice.
/// * `ends` - The ending indices for each dimension of the slice.
/// * `dims` - The number of dimensions to consider in the slice.
///
/// # Returns
///
/// * `Vec<f32>` - The data of the sliced tensor.
pub fn recursive_slice(
    tensor_data: &[f32],
    tensor_shape: &[usize],
    tensor_strides: &[usize],
    starts: &[usize],
    ends: &[usize],
    dims: usize,
) -> Vec<f32> {
    if dims == 1 {
        let mut result = Vec::with_capacity(ends[0] - starts[0]);
        for i in starts[0]..ends[0] {
            let flat_idx = i * tensor_strides[0];
            result.push(tensor_data[flat_idx]);
        }
        return result;
    }

    let mut result = Vec::new();
    for i in starts[0]..ends[0] {
        let sub_tensor_start = i * tensor_strides[0];
        let sub_tensor_end = sub_tensor_start + tensor_shape[1..].iter().product::<usize>();
        let sub_tensor_data = &tensor_data[sub_tensor_start..sub_tensor_end];
        result.extend(recursive_slice(
            sub_tensor_data,
            &tensor_shape[1..],
            &tensor_strides[1..],
            &starts[1..],
            &ends[1..],
            dims - 1,
        ))
    }
    result
}

pub fn are_broadcast_compatible(shape1: &[usize], shape2: &[usize]) -> bool {
    let s1_len = shape1.len();
    let s2_len = shape2.len();
    for i in 0..max(s1_len, s2_len) {
        let dim1 = if i >= s1_len {
            &1
        } else {
            &shape1[s1_len - 1 - i]
        };
        let dim2 = if i >= s2_len {
            &1
        } else {
            &shape2[s2_len - 1 - i]
        };

        if *dim1 != 1 && *dim2 != 1 && dim1 != dim2 {
            return false;
        }
    }

    true
}

pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    let s1_len = shape1.len();
    let s2_len = shape2.len();
    let mut result = Vec::with_capacity(max(s1_len, s2_len));
    for i in 0..result.capacity() {
        let dim1 = if i >= s1_len {
            1
        } else {
            shape1[s1_len - 1 - i]
        };
        let dim2 = if i >= s2_len {
            1
        } else {
            shape2[s2_len - 1 - i]
        };
        result.push(max(dim1, dim2));
    }

    result.reverse();
    result
}

// TODO: Maybe move to iterators module

pub struct MultiDimIter {
    shape: Vec<usize>,
    current: Vec<usize>,
    started: bool,
}

impl Iterator for MultiDimIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.started {
            self.started = true;
            return Some(self.current.clone());
        }

        for i in (0..self.shape.len()).rev() {
            if self.current[i] + 1 < self.shape[i] {
                self.current[i] += 1;

                for j in i + 1..self.shape.len() {
                    self.current[j] = 0;
                }

                return Some(self.current.clone());
            }
        }

        None
    }
}

pub fn multi_dim_iter(shape: &[usize]) -> MultiDimIter {
    MultiDimIter {
        shape: shape.to_vec(),
        current: vec![0; shape.len()],
        started: false,
    }
}
