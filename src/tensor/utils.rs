pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..(shape.len() - 1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1];
    }
    strides
}

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
