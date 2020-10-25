mod utils;

use wasm_bindgen::prelude::*;
use rust_nn::{Tensor, mse_loss};

// When the `wee_alloc` feature is enabled, use `wee_alloc` as the global
// allocator.
#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
extern {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet() {
    let targets = Tensor::ones(&[5]);
    let preds = Tensor::from_vec(vec![0.0,1.0,2.0,3.0,4.0]).unwrap();
    let l = mse_loss(&preds, &targets).unwrap();
    alert(&format!("Hello, rust-wasm-nn!, The loss is {}", l.loss));
}
