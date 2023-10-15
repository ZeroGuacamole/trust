pub trait Sum {
    fn sum(&self, dim: Option<usize>) -> Self;
}

pub trait Average {
    fn mean(&self, dim: Option<usize>) -> Self;
}

pub trait Max {
    fn max(&self, dim: Option<usize>) -> Self;
}

pub trait Min {
    fn min(&self, dim: Option<usize>) -> Self;
}
