use serde::{Deserialize, Serialize};

pub mod messages;
pub mod experiment;
pub mod constants;
pub mod time;

pub use experiment::*;
pub use messages::*;

#[derive(Copy, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceId {
    RoadsideUnit(u32),
    ZoneProcessor(u32),
}
