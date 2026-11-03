//! Durable state: store contracts, implementations, and schema.
//!
//! Everything that answers "where is this kept": the async storage
//! traits, their in-memory implementations, versioned schema migrations,
//! and the operator-only session retention contract.

pub mod data_lifecycle;
pub mod in_memory_stores;
pub mod migration;
pub mod store;
