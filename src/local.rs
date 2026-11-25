//! Compatibility facade and matrix-free, scalable unconstrained solvers.
//!
//! The legacy methods remain available unchanged. New algorithms use typed options.
//!
//! Prefer [`lbfgs()`] for gradient-only problems when dense storage is undesirable.
//! Implement [`HessianVector`] for [`newton_cg()`] or [`trust_region_cg`]. These
//! entry points minimize without bounds and return [`crate::Solution`].
//! [`strong_wolfe`] and [`steihaug()`] are reusable subproblem utilities, not full
//! minimizers. See [`crate::bounded`] for feasible box-constrained gradients.
pub use crate::derivative_free;
pub use crate::{Method, Options, minimize};

pub mod lbfgs;
pub mod newton_cg;
pub mod steihaug;
pub mod wolfe;

pub use lbfgs::{LbfgsOptions, lbfgs};
pub use newton_cg::{HessianVector, NewtonCgOptions, newton_cg};
pub use steihaug::{
    SteihaugOptions, SteihaugStep, SteihaugTermination, TrustRegionCgOptions, steihaug,
    trust_region_cg,
};
pub use wolfe::{StrongWolfeOptions, StrongWolfeStep, strong_wolfe};
