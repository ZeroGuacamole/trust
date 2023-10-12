pub trait Add {
    fn add(&self, other: &Self) -> Result<Box<Self>, String>;
}

pub trait Sub {
    fn sub(&self, other: &Self) -> Result<Box<Self>, String>;
}

pub trait Mul {
    fn mul(&self, other: &Self) -> Result<Box<Self>, String>;
}

pub trait Div {
    fn div(&self, other: &Self) -> Result<Box<Self>, String>;
}
