/*
    Notes:
    - Storing the data as 1D for memory management simplification
    and efficient utilization of cache
    - Shape will determine the dimensions of the tendor
    - Strides for conmputing the position of an element in flat array
    based on its multi-dimensional index
*/
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
}

fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for idx in (0..(shape.len() - 1)).rev() {
        strides[idx] = strides[idx + 1] * shape[idx + 1];
    }
    strides
}
