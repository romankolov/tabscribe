//! Tablature: the assignment of fingerings to notes and the cost model that
//! drives it.

pub mod assign;
pub mod build;
pub mod cost;
pub mod shapes;

pub use assign::{Assignment, Group, TabOptimizer};
pub use build::TabBuilder;
pub use cost::{CostWeights, Position, Style};
pub use shapes::ShapeDictionary;
