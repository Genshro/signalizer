//! RIBQA (Ribqa Intelligent Behavioral Quantitative Analysis) Module
//!
//! Bu modül RIBQA algoritmasını modüler bir yapıda organize eder:
//! - config: Konfigürasyon ve ayarlar
//! - types: Veri tipleri ve enum'lar
//! - analyzer: Ana analiz motoru
//! - signals: Sinyal üretim sistemi
//! - calculations: Matematiksel hesaplamalar

pub mod analyzer;
pub mod calculations;
pub mod config;
pub mod signals;
pub mod types;

// Re-export ana bileşenler
pub use analyzer::RibqaAnalyzer;
pub use calculations::{FractalCalculator, RecurrenceCalculator, TurbulenceCalculator};
pub use config::RibqaConfig;
pub use signals::RibqaSignalGenerator;
pub use types::{MarketRegime, RibqaResult};
