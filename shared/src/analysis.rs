use crate::types::ObjectCounts;

#[derive(Debug)]
pub enum CongestionLevel {
    Clear,
    Light,
    Moderate,
    Severe,
}

#[derive(Debug)]
pub struct IntersectionStatus {
    pub congestion_level: CongestionLevel,
    pub heavy_vehicle_impact: bool,
    pub pedestrian_safety_concern: bool,
    pub cyclist_present: bool,
    pub intervention_needed: bool,
}

pub fn analyze_congestion(counts: &ObjectCounts) -> String {
    let total_vehicles = counts.total_vehicles;

    if total_vehicles >= 15 {
        "severe".to_string()
    } else if total_vehicles >= 10 {
        "moderate".to_string()
    } else if total_vehicles >= 5 {
        "light".to_string()
    } else {
        "clear".to_string()
    }
}

pub fn analyze_intersection_congestion(counts: &ObjectCounts) -> IntersectionStatus {
    let heavy_vehicles = counts.trucks + counts.buses;
    let total_vehicles = counts.total_vehicles;

    IntersectionStatus {
        congestion_level: match total_vehicles {
            0..=4 => CongestionLevel::Clear,
            5..=9 => CongestionLevel::Light,
            10..=14 => CongestionLevel::Moderate,
            _ => CongestionLevel::Severe,
        },
        heavy_vehicle_impact: heavy_vehicles > 2,
        pedestrian_safety_concern: counts.pedestrians > 3 && total_vehicles > 8,
        cyclist_present: counts.bicycles > 0,
        intervention_needed: total_vehicles > 12 || (heavy_vehicles > 3),
    }
}